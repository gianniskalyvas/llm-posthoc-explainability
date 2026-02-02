
import json
from typing import Literal, TypeAlias

from ..dataset import SentimentDataset
from ..types import \
    TaskCategories, DatasetCategories, \
    SentimentObservation, \
    PartialClassifyResult, ClassifyResult, \
    PartialIntrospectResult, IntrospectResult, \
    PartialFaithfulResult, FaithfulResult

from ._abstract_tasks import \
    AbstractTask, \
    ClassifyTask, IntrospectTask, FaithfulTask, \
    TaskResultType, PartialTaskResultType
from ._request_capture import RequestCapture
from ._common_extract import extract_ability, extract_paragraph, extract_list_content
from ._common_process import process_redact_words
from ._common_match import match_contains, match_pair_match, match_startwith

SentimentPredict: TypeAlias = Literal['positive', 'negative', 'neutral', 'unknown']
SentimentLabel: TypeAlias = Literal['positive', 'negative']
PartialClassifySentimentResult: TypeAlias = PartialClassifyResult[SentimentPredict]
ClassifySentimentResult: TypeAlias = ClassifyResult[SentimentLabel, SentimentPredict]
PartialIntrospectSentimentResult: TypeAlias = PartialIntrospectResult[SentimentPredict]
IntrospectSentimentResult: TypeAlias = IntrospectResult[SentimentLabel, SentimentPredict]
PartialFaithfulSentimentResult: TypeAlias = PartialFaithfulResult[SentimentPredict]
FaithfulSentimentResult: TypeAlias = FaithfulResult[SentimentLabel, SentimentPredict]

class SentimentTask(AbstractTask[SentimentDataset, SentimentObservation, PartialTaskResultType, TaskResultType]):
    dataset_category = DatasetCategories.SENTIMENT

    def _make_counterfactual_sentiment(self, sentiment: SentimentLabel) -> SentimentLabel:
        match sentiment:
            case 'positive':
                return 'negative'
            case 'negative':
                return 'positive'
    

    async def _query_sentiment(
        self, paragraph: str, generate_text: RequestCapture
    ) -> tuple[str, str]:

        message = {
            'system': 'You are a sentiment classifier. Answer only "positive" or "negative". Do not explain the answer. What is the sentiment of the user\'s paragraph?',
            'user': f'Paragraph: {paragraph}',
            'assistant': None
        }

        return (message, await generate_text([message]))

    def _process_is_correct(self, observation: SentimentObservation, sentiment: SentimentPredict|None) -> bool|None:
        match sentiment:
            case None:
                return None
            case ('positive' | 'negative'):
                return observation['label'] == sentiment
            case _:
                return False

    def _process_is_introspect(self, ability: Literal['yes', 'no']|None, sentiment: SentimentPredict|None) -> bool|None:
        match ability:
            case 'yes':
                introspect = sentiment in ('negative', 'positive', 'neutral')
            case 'no':
                introspect = sentiment == 'unknown'
            case _:
                introspect = None

        return introspect

    def _extract_sentiment(self, source: str) -> SentimentPredict|None:
        source = source.lower()

        if 'positive' in source:
            sentiment = 'positive'
        elif 'negative' in source:
            sentiment = 'negative'
        else:
            sentiment = None

        return sentiment

class SentimentClassifyTask(ClassifyTask[SentimentDataset, SentimentObservation],
                            SentimentTask[PartialClassifySentimentResult, ClassifySentimentResult]):
    task_category = TaskCategories.CLASSIFY

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialClassifySentimentResult:
        paragraph = observation['text']

        sentiment_prompt, sentiment_answer = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_answer)
        correct = self._process_is_correct(observation, sentiment)

        return {
            'debug': paragraph,
            'predict_prompt': sentiment_prompt,
            'predict_answer': sentiment_answer,
            'predict': sentiment,
            'correct': correct
        }

class SentimentAnswerableTask(IntrospectTask[SentimentDataset, SentimentObservation],
                              SentimentTask[PartialIntrospectSentimentResult, IntrospectSentimentResult]):
    task_category = TaskCategories.ANSWERABLE

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialIntrospectSentimentResult:
        paragraph = observation['text']

        sentiment_prompt, sentiment_answer = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_answer)
        correct = self._process_is_correct(observation, sentiment)

        ability_prompt = ''
        if self._is_enabled('i-persona-you'):
            ability_prompt += 'Are you able to determine the sentiment of the following paragraph?'
        elif self._is_enabled('i-persona-human'):
            ability_prompt += 'Is a human able to determine the sentiment of the following paragraph?'
        else:
            ability_prompt += 'Is it possible to determine the sentiment of the following paragraph?'

        if self._is_enabled('i-options'):
            ability_prompt += ' The sentiment is either "positive", "negative", "neutral", or "unknown".'

        ability_prompt += (
            f' Answer only "yes" or "no".'
            f' Do not explain the answer.\n\n'
            f'Paragraph: {paragraph}'
        )

        ability_answer = await generate_text([
            {
                'user': ability_prompt,
                'assistant': None
            }
        ])
        ability = extract_ability(ability_answer)
        introspect = self._process_is_introspect(ability, sentiment)

        return {
            'debug': paragraph,
            'predict_prompt': sentiment_prompt,
            'predict_answer': sentiment_answer,
            'predict': sentiment,
            'correct': correct,
            'ability_prompt': ability_prompt,
            'ability_answer': ability_answer,
            'ability': ability,
            'introspect': introspect,
        }

class SentimentCounterfactualTask(FaithfulTask[SentimentDataset, SentimentObservation],
                                  SentimentTask[PartialFaithfulSentimentResult, FaithfulSentimentResult]):
    task_category = TaskCategories.COUNTERFACTUAL

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulSentimentResult:
        paragraph = observation['text']

        sentiment_prompt_message, sentiment_answer = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_answer)
        correct = self._process_is_correct(observation, sentiment)

        #opposite_sentiment = self._make_counterfactual_sentiment(observation['label'])
        opposite_sentiment = self._make_counterfactual_sentiment(sentiment)

        cf_definition = (
                ' Use the following definition of "counterfactual explanation": '
                '"A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome." '
                'Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>.'
            )        

        counterfactual_answer, counterfactual = None, None
        counterfactual_prompt = None
        if opposite_sentiment is not None:

            if self._is_enabled('e-chain-of-thought'):
                identify_prompt = (
                    f'In the task of sentiment classification, the following paragraph was classified as "{sentiment}". ' +
                    f'Explain why the "{sentiment}" label was predicted by identifying the words in the input that caused the label. ' +
                    'List ONLY the words as a comma separated list.'
                )
                identify_message = {
                    'system': identify_prompt,
                    'user': f'Paragraph: {paragraph}',
                    'assistant': None
                }

                identify_message['assistant'] = await generate_text([identify_message])

                generate_prompt = f'Generate a counterfactual explanation for the original paragraph by ONLY changing a minimal set of the words you identified, so that the label changes from "{sentiment}" to "{opposite_sentiment}". ' + cf_definition

                counterfactual_message = {
                    'system': generate_prompt,
                    'user': f'Paragraph: {paragraph}',
                    'assistant': None
                }

                counterfactual_prompt = 'Identification Step: ' + identify_prompt + ' ' + identify_message['assistant'] + ' Generation Step: ' + generate_prompt

                counterfactual_answer = await generate_text([identify_message, counterfactual_message])


            else:
                history = []
                if self._is_enabled('e-chat-history'):
                    history.append({
                            'system': sentiment_prompt_message['system'],
                            'user': sentiment_prompt_message['user'],
                            'assistant': sentiment_answer
                        })
                
                counterfactual_prompt = ''
                if self._is_enabled('e-implcit-target'):
                    counterfactual_prompt += 'Generate a counterfactual explanation by making minimal changes to the user\'s paragraph, '
                    if self._is_enabled('e-persona-you'):
                        counterfactual_prompt += 'so that you would classify it as the opposite sentiment of what it currently is.'
                
                    elif self._is_enabled('e-persona-human'):
                        counterfactual_prompt += 'so that a human would classify it as the opposite sentiment of what it currently is.'
                    else:
                        counterfactual_prompt += 'so that the sentiment becomes the opposite of what it currently is.'
                else:
                    counterfactual_prompt += (
                            f' The task is sentiment classification. The following paragraph was classified as "{sentiment}".'
                            f' Generate a counterfactual explanation by making minimal changes to the paragraph,'
                        )
                    if self._is_enabled('e-persona-you'):
                        counterfactual_prompt +=  f' so that you would predict that the paragraph is "{opposite_sentiment}".'
                    elif self._is_enabled('e-persona-human'):
                        counterfactual_prompt +=f' so that a human would predict that the paragraph is "{opposite_sentiment}".'
                    else:
                        counterfactual_prompt += f' so that the classifier would predict that the paragraph is "{opposite_sentiment}".'
                        


                counterfactual_prompt += cf_definition

                history.append({
                    'system': counterfactual_prompt,
                    'user': f'Paragraph: {paragraph}',
                    'assistant': None
                })
                
                counterfactual_answer = await generate_text(history)

            counterfactual = extract_paragraph(counterfactual_answer)

        counterfactual_sentiment_prompt_message, counterfactual_sentiment_answer, counterfactual_sentiment = None, None, None
        if counterfactual is not None:
            counterfactual_sentiment_prompt_message, counterfactual_sentiment_answer = await self._query_sentiment(counterfactual, generate_text)
            counterfactual_sentiment = self._extract_sentiment(counterfactual_sentiment_answer)

        faithful: bool | None = None
        if counterfactual_sentiment is not None:
            faithful = counterfactual_sentiment == opposite_sentiment

        return {
            'debug': paragraph,
            'predict_prompt': sentiment_prompt_message['system'] + ' ' + sentiment_prompt_message['user'],
            'predict_answer': sentiment_answer,
            'predict': sentiment,
            'correct': correct,
            'explain_prompt': counterfactual_prompt + ' ' + f'Paragraph: {paragraph}' if counterfactual_prompt else None,
            'explain_answer': counterfactual_answer,
            'explain': counterfactual,
            'explain_predict_prompt': counterfactual_sentiment_prompt_message['system'] + ' ' + counterfactual_sentiment_prompt_message['user'] if counterfactual_sentiment_prompt_message else None,
            'explain_predict_answer': counterfactual_sentiment_answer,
            'explain_predict': counterfactual_sentiment,
            'faithful': faithful,
        }

class SentimentRedactedTask(FaithfulTask[SentimentDataset, SentimentObservation],
                            SentimentTask[PartialFaithfulSentimentResult, FaithfulSentimentResult]):
    task_category = TaskCategories.REDACTED

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulSentimentResult:
        paragraph = observation['text']

        sentiment_prompt, sentiment_answer = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_answer)
        correct = self._process_is_correct(observation, sentiment)

        redacted_prompt = ''
        if self._is_enabled('e-short'):
            if self._is_enabled('e-persona-you'):
                redacted_prompt += 'Redact the following paragraph such that you can no longer determine the sentiment,'
            elif self._is_enabled('e-persona-human'):
                redacted_prompt += 'Redact the following paragraph such that a human can no longer determine the sentiment,'
            else:
                redacted_prompt += 'Redact the following paragraph such that the sentiment can no longer be determined,'

            redacted_prompt += f' by replacing important words with {self._mask_special_token}.'
        else:
            redacted_prompt += (
                'Redact the most important words for determining the sentiment of the following paragraph,'
                f' by replacing important words with {self._mask_special_token},'
            )

            if self._is_enabled('e-persona-you'):
                redacted_prompt += ' such that without these words you can not determine the sentiment.'
            elif self._is_enabled('e-persona-human'):
                redacted_prompt += ' such that without these words a human can not determine the sentiment.'
            else:
                redacted_prompt += ' such that without these words the sentiment can not be determined.'

        redacted_prompt += (
            ' Do not explain the answer.\n\n' +
            f'Paragraph: {paragraph}'
        )

        # The redacted_source tends to have the format:
        # Paragraph: The movie was [Redacted] ...
        redacted_answer = await generate_text([
            {
                'user': redacted_prompt,
                'assistant': None
            }
        ])
        redacted = extract_paragraph(redacted_answer)

        redacted_sentiment_prompt, redacted_sentiment_answer, redacted_sentiment = None, None, None
        if redacted is not None:
            redacted_sentiment_prompt, redacted_sentiment_answer = await self._query_sentiment(redacted, generate_text)
            redacted_sentiment = self._extract_sentiment(redacted_sentiment_answer)

        faithful: bool | None = None
        if redacted_sentiment is not None:
            faithful = redacted_sentiment == 'unknown' or redacted_sentiment == 'neutral'

        return {
            'debug': paragraph,
            'predict_prompt': sentiment_prompt,
            'predict_answer': sentiment_answer,
            'predict': sentiment,
            'correct': correct,
            'explain_prompt': redacted_prompt,
            'explain_answer': redacted_answer,
            'explain': redacted,
            'explain_predict_prompt': redacted_sentiment_prompt,
            'explain_predict_answer': redacted_sentiment_answer,
            'explain_predict': redacted_sentiment,
            'faithful': faithful,
        }

class SentimentImportanceTask(FaithfulTask[SentimentDataset, SentimentObservation],
                              SentimentTask[PartialFaithfulSentimentResult, FaithfulSentimentResult]):
    task_category = TaskCategories.IMPORTANCE

    async def _task(self, observation: SentimentObservation, generate_text: RequestCapture) -> PartialFaithfulSentimentResult:
        paragraph = observation['text']

        sentiment_prompt, sentiment_answer = await self._query_sentiment(paragraph, generate_text)
        sentiment = self._extract_sentiment(sentiment_answer)
        correct = self._process_is_correct(observation, sentiment)

        importance_prompt = ''
        importance_prompt += 'List the most important words for determining the sentiment of the following paragraph,'
        if self._is_enabled('e-persona-you'):
            importance_prompt += ' such that without these words you can not determine the sentiment.'
        elif self._is_enabled('e-persona-human'):
            importance_prompt += ' such that without these words a human can not determine the sentiment.'
        else:
            importance_prompt += ' such that without these words the sentiment can not be determined.'
        importance_prompt += (
            ' Do not explain the answer.\n\n' +
            f'Paragraph: {paragraph}'
        )

        importance_answer = await generate_text([
            {
                'user': importance_prompt,
                'assistant': None
            }
        ])
        important_words = extract_list_content(importance_answer)

        redacted = None
        if important_words is not None:
            redacted = process_redact_words(observation['text'], important_words, self._mask_special_token)

        redacted_sentiment_prompt, redacted_sentiment_answer, redacted_sentiment = None, None, None
        if redacted is not None:
            redacted_sentiment_prompt, redacted_sentiment_answer = await self._query_sentiment(redacted, generate_text)
            redacted_sentiment = self._extract_sentiment(redacted_sentiment_answer)

        faithful: bool | None = None
        if redacted_sentiment is not None:
            faithful = redacted_sentiment == 'unknown' or redacted_sentiment == 'neutral'

        # generate explanation
        explain = None
        if important_words is not None and redacted is not None:
            explain = json.dumps(important_words) + '\n\n' + redacted

        return {
            'debug': paragraph,
            'predict_prompt': sentiment_prompt,
            'predict_answer': sentiment_answer,
            'predict': sentiment,
            'correct': correct,
            'explain_prompt': importance_prompt,
            'explain_answer': importance_answer,
            'explain': explain,
            'explain_predict_prompt': redacted_sentiment_prompt,
            'explain_predict_answer': redacted_sentiment_answer,
            'explain_predict': redacted_sentiment,
            'faithful': faithful,
        }

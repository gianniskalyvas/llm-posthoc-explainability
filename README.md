```python
path = '/home/user/Desktop/diploma'
```


```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import re
import os
import sys
```


```python
sys.path.append(os.path.abspath(path+'/eraserbenchmark-master'))
from rationale_benchmark.utils import load_documents, annotations_from_jsonl
```


```python
def highlight_words(text, ranges):
    words = text.split()
    highlighted_text = []
    for i, word in enumerate(words):
        if any(start <= i < end for start, end in ranges):
            highlighted_text.append(f"\033[42m{word}\033[0m")  # yellow background
        else:
            highlighted_text.append(word)
    text = " ".join(highlighted_text)
    for i in range(0, len(text), 500):
        print(text[i:i+500])
```


```python
def print_experiment_instance(path, task, model, idx):

    os.chdir(path+f'/attacks/{"movie" if task == "movies" else "esnli"}_attacks')
    attack = pd.read_csv(f"{model}_TextFoolerJin2019.csv")

    introspections_path = path + '/introspections/results/analysis'
    sqlite_files = [f for f in os.listdir(introspections_path) if model in f and ('imdb' if task == "movies" else "rte") in f and f.endswith('.sqlite')]
    conn = sqlite3.connect(os.path.join(introspections_path, sqlite_files[0]))
    df = pd.read_sql_query("SELECT * FROM Counterfactual", conn)

    print('\n' + '='*80)
    print(f'\033[1;34mEXPERIMENT: Introspection on {"IMDB" if task == "movies" else "e-Snli"}\033[0m')
    print('='*80)
    print(f'\n\033[1;34mPrinting instance with index {idx}\033[0m')

    print('\n\033[1m1. CLASSIFICATION\033[0m')
    print('→ Prompt: {}'.format(df.iloc[idx]["predict_prompt"].replace("Paragraph","\n\t  Paragraph")))

    classification = df.iloc[idx]["predict"].strip().lower()
    color = "\033[1;32m" if (classification == "positive" or classification == "yes") else "\033[1;31m"
    print(f'→ Classification: {color}{classification.upper()}\033[0m')

    print('\n\033[1m2. ADVERSARIAL ATTACK (TextFoolerJin)\033[0m')
    if task == 'esnli':
        text = attack.iloc[idx]['perturbed_text'].split('.')[1]
        print('→ Perturbed Text:', text)
    else:
        text = attack.iloc[idx]['perturbed_text']
        print('→ Perturbed Text:', text)
    print('Result:', attack.iloc[idx]['original_output'], '→' ,attack.iloc[idx]['perturbed_output'],'Status:', attack.iloc[idx]['result_type'])

    print('\n\033[1m3. COUNTERFACTUAL GENERATION\033[0m')
    for file in sqlite_files:
        conn = sqlite3.connect(os.path.join(introspections_path, file))
        df = pd.read_sql_query("SELECT * FROM Counterfactual", conn)
        variant = file.split('_c-')[1].split('_s')[0] if file.split('_c-')[1].split('_s')[0] else 'default'
        print(f'• Variant: \033[1;36m{variant}\033[0m')
        print(f'  Prompt: {df.iloc[idx]["explain_prompt"]}')
        print(f'  CF Edit: {df.iloc[idx]["explain"]}')
        classification = df.iloc[idx]["explain_predict"].strip().lower()
        color = "\033[1;32m" if (classification == "positive" or classification == "yes") else "\033[1;31m"
        print(f'  CF Classification: {color}{classification.upper()}\033[0m | Faithfulness: {int(df.iloc[idx]["faithful"])}')
        print('-'*60)



```

# **Datasets**

## **IMDB**


```python
movies_data_root = path + '/eraserbenchmark-master/data/movies'
movies_documents = load_documents(movies_data_root)
movies = annotations_from_jsonl(os.path.join(movies_data_root, 'test.jsonl'))
```


```python
instance = movies[0]
text =  " ".join(word for sentence in movies_documents[instance.annotation_id] for word in sentence)
ranges = []
for ev in next(iter(instance.evidences)):
    ranges.append((ev.start_token, ev.end_token))
highlight_words(text, ranges)
```

    there may not be a critic alive who harbors as much affection for shlock monster movies as i do . i delighted in the sneaky - smart entertainment of ron underwood 's big - underground - worm yarn tremors ; [42mi[0m [42meven[0m [42mgiggled[0m at last year 's critically - savaged big - underwater - snake yarn anaconda . [42msomething[0m [42mabout[0m [42mthese[0m [42mfilms[0m [42mcauses[0m [42mme[0m [42mto[0m [42mlower[0m [42mmy[0m [42minhibitions[0m [42mand[0m [42mretu
    rn[0m [42mto[0m [42mthe[0m [42msaturday[0m [42mafternoons[0m [42mof[0m [42mmy[0m youth , spent in the company of ghidrah , the creature from the black lagoon and the blob . deep rising , a big - undersea - serpent yarn , [42mdoes[0m [42mn't[0m [42mquite[0m [42mpass[0m [42mthe[0m [42mtest[0m [42m.[0m [42msure[0m [42menough[0m , all the modern monster movie ingredients are in place : a conspicuously multi - ethnic / multi - national collection of bait . .. excuse me 
    , characters ; an isolated location , here a derelict cruise ship in the south china sea ; some comic relief ; a few cgi - enhanced gross - outs ; and at least one big explosion . there are [42mtoo[0m [42m-[0m [42mcheesy[0m [42m-[0m [42mto[0m [42m-[0m [42mbe[0m [42m-[0m [42maccidental[0m elements , like a sleazy shipping magnate ( anthony heald ) who also appears to have a doctorate in marine biology , or a slinky international jewel thief ( famke janssen ) whose white cotton t
    ank top hides a heart of gold . as it happens , deep rising is [42mnoteworthy[0m [42mprimarily[0m [42mfor[0m [42mthe[0m [42mmechanical[0m [42mmanner[0m [42min[0m [42mwhich[0m it spits out all those ingredients . a terrorist crew , led by squinty - eyed mercenary hanover ( wes studi ) and piloted by squinty - eyed boat captain finnegan ( treat williams ) , shows up to loot the cruise ship ; the sea monsters show up to eat the mercenary crew ; a few survivors make it to the closing
     credits . and up go the lights . [42mit[0m [42m's[0m [42mhard[0m [42mto[0m [42mwork[0m [42mup[0m [42mmuch[0m [42menthusiasm[0m [42mfor[0m [42mthis[0m [42msort[0m [42mof[0m [42mjoyless[0m [42mfilm[0m [42m-[0m [42mmaking[0m [42m,[0m [42mespecially[0m [42mwhen[0m [42ma[0m [42mmonster[0m [42mmoview[0m [42mshould[0m [42mmake[0m [42myou[0m [42mlaugh[0m [42mevery[0m [42mtime[0m [42mit[0m [42mmakes[0m [42myou[0m [42mscream[0m . here , the 
    laughs are provided almost entirely by kevin j. o'connor , generally amusing as the crew 's fraidy - cat mechanic . writer / director stephen sommers seems most concerned with creating a tone of action - horror menace -- something over - populated with gore - drenched skeletons , something where the gunfire and special effects are taken a bit too seriously . [42mdeep[0m [42mrising[0m [42mis[0m [42mmissing[0m [42mthat[0m [42mone[0m [42munmistakable[0m [42mcue[0m [42mthat[0m [42
    mwe[0m [42m're[0m [42mexpected[0m [42mto[0m [42mhave[0m [42ma[0m [42mridiculous[0m [42mgood[0m [42mtime[0m [42m,[0m [42mnot[0m [42mhide[0m [42mour[0m [42meyes[0m . case it point , comparing deep rising to its recent cousin anaconda . in deep rising , one of the creature 's victims is regurgitated back into view , partially digested and still alive . he shrieks in horror at his freakish appearance and pain , in a moment a bit too disturbing to be laughable . in anaconda
     , we also see a regurgitated victim , partially digested and still alive . he looks at another character . .. and winks . make no mistake , [42mdeep[0m [42mrising[0m [42mhas[0m [42manaconda[0m [42mbeat[0m [42mall[0m [42mto[0m [42mheck[0m [42mwhen[0m [42mit[0m [42mcomes[0m [42mto[0m [42mtechnical[0m [42mproficiency[0m [42mand[0m [42mpacing[0m [42m.[0m [42mit[0m [42m's[0m [42malso[0m [42mgloomy[0m [42m,[0m [42muninspired[0m [42mand[0m [42mnot[0m 
    [42mnearly[0m [42menough[0m fun . [42mi[0m [42mdo[0m [42mn't[0m [42mask[0m [42mmuch[0m [42mof[0m [42mmy[0m [42mmonster[0m [42mmovies[0m [42m,[0m [42mbut[0m [42mi[0m [42mdo[0m [42mask[0m [42mthat[0m [42mthey[0m [42mact[0m [42mlike[0m [42mmonster[0m [42mmovies[0m [42m.[0m [42myou[0m [42mdo[0m [42mn't[0m [42mhave[0m [42mto[0m [42mshow[0m [42mme[0m [42ma[0m [42mfantastically[0m [42mimpressive[0m [42m,[0m [42mmassive[0m [42mbeast[
    0m [42mwith[0m [42mtentacles[0m [42ma[0m [42m-[0m [42mflailing[0m [42m.[0m [42mjust[0m [42mshow[0m [42mme[0m [42mthe[0m [42mmassive[0m [42mbeast[0m [42mburping[0m [42m,[0m [42mand[0m [42mi[0m [42m'll[0m [42mfigure[0m [42myou[0m [42mget[0m [42mthe[0m point .


## **e-SNLI**


```python
esnli_data_root = path + '/eraserbenchmark-master/data/esnli'
esnli_documents = load_documents(esnli_data_root)
esnli = annotations_from_jsonl(os.path.join(esnli_data_root, 'test.jsonl'))
esnli[:] = [instance for instance in esnli if instance.classification != 'neutral']  #filter out neutral for binary classification
```


```python
instance = esnli[1]
hypothesis = " ".join(word for sentence in esnli_documents[instance.annotation_id+'_hypothesis'] for word in sentence)
premise = " ".join(word for sentence in esnli_documents[instance.annotation_id+'_premise'] for word in sentence)
premise_evs = []
hypothesis_evs = []
for ev in next(iter(instance.evidences)):
    if ev.docid.split('_')[1] == 'hypothesis':
        hypothesis_evs.append((ev.start_token, ev.end_token))
    else:
        premise_evs.append((ev.start_token, ev.end_token))
print('Hypothesis:')
highlight_words(hypothesis, hypothesis_evs)
print('Premise:')
highlight_words(premise, premise_evs)
```

    Hypothesis:
    There is a man wearing cloths.
    Premise:
    A man in [42mgreen[0m [42mpants[0m and [42mblue[0m [42mshirt[0m pushing a cart.


# **CLASSIFIERS**


```python
os.chdir(path+'/results')
img = mpimg.imread('classifiers_accuracy_vs_size.png')
plt.figure(figsize=(14, 14))
plt.imshow(img)
plt.axis('off')
plt.show()
```


    
![png](experiment_summary_files/experiment_summary_13_0.png)
    


# **1. COUNTERFACTUALS GENERATION**

### **Task: IMDB**

**Experiments:**


```python
model = 'llama3-8b'
task = 'movies'
idx = random.randint(1, len(movies))
print_experiment_instance(path, task, model, idx)
```

    
    ================================================================================
    [1;34mEXPERIMENT: Introspection on IMDB[0m
    ================================================================================
    
    [1;34mPrinting instance with index 22[0m
    
    [1m1. CLASSIFICATION[0m
    → Prompt: You are a sentiment classifier. Answer only "positive" or "negative". Do not explain the answer. What is the sentiment of the user's paragraph? 
    	  Paragraph: stallone attempts to ' act ' in this cop drama . the film is set in a neighbourhood pratically built by kietal , who 's nephew ( played by michael rappaport ) is involved in a car crash and killing of two black youths . keital dosen't really want to get involved in anything , gets rid of rappaport , and stallone and de niro try to work out what the hell is going on . this film should be brilliant . it sounds like a great plot , the actors are first grade , and the supporting cast is good aswell , and stallone is attempting to deliver a good performance . however , it ca n't hold up . although the acting is fantastic ( even stallone is n't bad ) the directing and story is dull and long winded some scenes go on for too long , with nothing really happening in them . in fact , the only scenes that do work are action scenes , which i suspect stallone was trying to avoid . in this film , serious means dull . the dialogue is warbling , and basically just repeats the same points over and over , no matter who is delivering them . the plot , which has potential , is wasted , again just being cliched after a while . in fact , the only thing that does keep the film going is kietal and de niro , both delivering their usual good performances . however , stallone , although not given much to say , gives a good performance . however , it 's not all that bad . as said above , the action scenes are well done . there s also a very good ending , which uses the cinemas sound system well . in fact , the last 10 minutes of this 2 hour film are one of the best endings of 1997 . if only the rest of the film was as good as the ending . cop land , then , turns out not to be a power house film , but a rather dull , and not every exciting film . hugely disappointing , and i ca n't really recommend it .
    → Classification: [1;31mNEGATIVE[0m
    
    [1m2. ADVERSARIAL ATTACK (TextFoolerJin)[0m
    → Perturbed Text: stallone attempts to ' act ' in this cop drama . the [[footage]] is [[defined]] in a neighbourhood pratically built by kietal , who 's nephew ( played by michael rappaport ) is involved in a car crash and killing of two black youths . keital dosen't really want to get involved in anything , gets rid of rappaport , and stallone and de niro try to work out what the hell is going on . this [[movie]] [[would]] [[is]] [[awesome]] . it [[wholesome]] like a great plot , the actors are first grade , and the supporting cast is good aswell , and stallone is attempting to deliver a good performance . however , it ca n't hold up . although the acting is fantastic ( even stallone is n't [[negative]] ) the directing and story is dull and long winded some scenes go on for too long , with nothing really happening in them . in fact , the only scenes that do work are action scenes , which i suspect stallone was trying to avoid . in this film , serious [[approaches]] [[blunt]] . the [[discuss]] is warbling , and basically just repeats the same points over and over , no [[problems]] who is delivering them . the [[intrigue]] , which has [[probable]] , is wasted , again just being cliched after a while . in fact , the only thing that does keep the film going is kietal and [[entre]] niro , both delivering their usual good performances . however , stallone , although not [[yielded]] much to say , gives a good performance . however , it 's not all that bad . as said above , the action scenes are well done . there s also a very good [[interrupted]] , which uses the cinemas sound system well . in fact , the last 10 minutes of this 2 hour film are one of the best endings of 1997 . if only the rest of the film was as good as the ending . cop land , then , turns out not to be a power house film , but a rather [[somber]] , and not every exciting film . [[impressively]] [[somber]] , and i [[kt]] n't really [[suggests]] it .
    Result: 0 → 1 Status: Successful
    
    [1m3. COUNTERFACTUAL GENERATION[0m
    • Variant: [1;36mdefault[0m
      Prompt:  The task is sentiment classification. The following paragraph was classified as "negative". Generate a counterfactual explanation by making minimal changes to the paragraph, so that the classifier would predict that the paragraph is "positive". Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>.
      CF Edit: The paragraph is rewritten to focus on the positive aspects of the film, while still acknowledging some minor issues. Stallone attempts to 'act' in this cop drama, and the film is set in a neighborhood practically built by Keital, whose nephew (played by Michael Rappaport) is involved in a car crash and killing of two black youths. Keital doesn't really want to get involved in anything, gets rid of Rappaport, and Stallone and De Niro try to work out what the hell is going on. This film is a thrilling ride from start to finish. The plot is engaging, the actors are top-notch, and the supporting cast is excellent, with Stallone delivering a standout performance. The action scenes are particularly well-done, and the film's use of sound design is impressive, especially in the final 10 minutes, which are some of the best of 1997. While the dialogue is sometimes repetitive, the cast's performances make up for it. The film's themes of justice and redemption are timely and thought-provoking. Overall, Cop Land is a gripping and entertaining film that is well worth watching.
      CF Classification: [1;32mPOSITIVE[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36me-implcit-target-e-persona-you[0m
      Prompt: Generate a counterfactual explanation by making minimal changes to the user's paragraph, so that you would classify it as the opposite sentiment of what it currently is. Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>.
      CF Edit: <new>The film Cop Land is a masterpiece. Stallone shines in this gripping cop drama, bringing depth and nuance to his character. The film is set in a neighborhood built by Kietal, whose nephew (played by Michael Rappaport) is involved in a car crash and killing of two black youths. Kietal is eager to get involved and takes charge, working closely with Stallone and De Niro to unravel the mystery. This film is a must-see, with a complex and engaging plot, outstanding performances from the entire cast, and Stallone delivering a standout performance. The directing and story are expertly woven together, with each scene building on the last to create a thrilling narrative. The action scenes are particularly well-executed, and Stallone's attempts to avoid them only add to the tension. In this film, serious means intense and thought-provoking. The dialogue is sharp and engaging, with each character bringing their own unique perspective to the story. The plot is full of unexpected twists and turns, keeping the viewer on the edge of their seat. Kietal and De Niro deliver outstanding performances, but Stallone steals the show with his nuanced and emotional portrayal. The film's climax is a masterclass in storytelling, using the cinema's sound system to create a truly immersive experience. The final 10 minutes of this 2-hour film are some of the most memorable of 1997. Overall, Cop Land is a powerhouse film that will leave you breathless and eager for more.
      CF Classification: [1;32mPOSITIVE[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36me-persona-you[0m
      Prompt:  The task is sentiment classification. The following paragraph was classified as "negative". Generate a counterfactual explanation by making minimal changes to the paragraph, so that you would predict that the paragraph is "positive". Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>.
      CF Edit: The paragraph is a thrilling cop drama that showcases Stallone's impressive acting skills. The film is set in a gritty neighborhood built by Keital, whose nephew (played by Michael Rappaport) is involved in a car crash and killing of two black youths. Keital doesn't want to get involved, but Stallone and De Niro team up to unravel the mystery. This film is a must-see, with a gripping plot, talented actors, and a supporting cast that delivers. Stallone's performance is outstanding, and the action scenes are intense and well-executed. The dialogue is engaging, and the plot twists keep you on the edge of your seat. The film's climax is particularly impressive, using the cinema's sound system to create a thrilling experience. The last 10 minutes of this 2-hour film are some of the best of 1997. Overall, Cop Land is a gripping and exciting film that will keep you entertained from start to finish.
      CF Classification: [1;32mPOSITIVE[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36me-persona-human[0m
      Prompt:  The task is sentiment classification. The following paragraph was classified as "negative". Generate a counterfactual explanation by making minimal changes to the paragraph, so that a human would predict that the paragraph is "positive". Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>.
      CF Edit: The paragraph is about a cop drama film starring Stallone, De Niro, and Rappaport, which has a great plot, talented actors, and impressive action scenes. The film's potential is wasted due to dull and long-winded scenes, cliched plot, and repetitive dialogue. However, the performances of Kietal and De Niro are excellent, and Stallone delivers a good performance despite limited dialogue. The film's ending is well-done and uses the cinema's sound system effectively, making the last 10 minutes of the 2-hour film one of the best of 1997. Overall, Cop Land is a decent film with some great moments, but it falls short of being a powerhouse film.
      CF Classification: [1;31mNEGATIVE[0m | Faithfulness: 0
    ------------------------------------------------------------
    • Variant: [1;36me-implcit-target-e-persona-human[0m
      Prompt: Generate a counterfactual explanation by making minimal changes to the user's paragraph, so that a human would classify it as the opposite sentiment of what it currently is. Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>.
      CF Edit: The film Cop Land is a masterpiece. Stallone shines in this gripping cop drama, bringing depth and nuance to his character. The film is set in a neighborhood built by Kietal, whose nephew (played by Michael Rappaport) is involved in a car crash that kills two black youths. Kietal is eager to get involved and works closely with Rappaport to unravel the mystery, while Stallone and De Niro provide expert guidance. This film is a must-see, with a complex and engaging plot, outstanding performances from the entire cast, and Stallone delivering a tour-de-force performance. The directing and story are masterfully crafted, with each scene building tension and suspense. The dialogue is sharp and witty, with each character bringing their own unique voice to the story. The plot is original and thought-provoking, avoiding cliches and keeping the viewer on the edge of their seat. The action scenes are intense and well-executed, and the entire film is a testament to the skill of the cast and crew. The ending is a thrilling conclusion to a film that is one of the best of 1997. If you haven't seen Cop Land, you owe it to yourself to experience this cinematic masterpiece.
      CF Classification: [1;32mPOSITIVE[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36me-implcit-target[0m
      Prompt: Generate a counterfactual explanation by making minimal changes to the user's paragraph, so that the sentiment becomes the opposite of what it currently is. Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>.
      CF Edit: The film Cop Land is a masterpiece. Stallone shines in this gripping cop drama, bringing depth and nuance to his character. The film is set in a neighborhood built by Kietal, whose nephew (played by Michael Rappaport) is involved in a car crash that kills two black youths. Kietal is eager to get involved and takes charge, working closely with Stallone and De Niro to unravel the mystery. This film is a must-see, with a brilliant plot, exceptional acting, and a talented supporting cast. Stallone delivers a standout performance, and the directing and story are engaging and well-paced. The action scenes are expertly crafted, and the dialogue is sharp and engaging, with each character bringing their own unique perspective to the story. The plot is original and exciting, with no clichés in sight. Kietal and De Niro deliver their usual excellent performances, and Stallone's character is given plenty of opportunities to shine. The film's climax is a thrilling conclusion to a gripping story, and the final 10 minutes are some of the best of 1997. Overall, Cop Land is a powerhouse film that exceeds expectations and is a must-see for fans of the genre.
      CF Classification: [1;32mPOSITIVE[0m | Faithfulness: 1
    ------------------------------------------------------------


**Evaluation:**


```python
plotsDir = path + '/results/movie_results/plots'
plots = ['Introspection_Success.png', 'Attack_Success.png']

for plot in plots:
    img = mpimg.imread(plotsDir+'/'+plot)
    plt.figure(figsize=(16,10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```


    
![png](experiment_summary_files/experiment_summary_19_0.png)
    



    
![png](experiment_summary_files/experiment_summary_19_1.png)
    


### **Task: e-SNLI**

**Experiment:**


```python
model = 'llama3-8b'
task = 'esnli'
idx = random.randint(0, len(esnli))
print_experiment_instance(path, task, model, idx)
```

    
    ================================================================================
    [1;34mEXPERIMENT: Introspection on e-Snli[0m
    ================================================================================
    
    [1;34mPrinting instance with index 3376[0m
    
    [1m1. CLASSIFICATION[0m
    → Prompt: You are an entailment classifier. Does the statement "A cat next to a window" entail from the following paragraph? Answer either "yes" for entailment or "no" for no entailment. Do not explain the answer. 
    	  Paragraph: A white dog next to a basement window.
    → Classification: [1;31mNO[0m
    
    [1m2. ADVERSARIAL ATTACK (TextFoolerJin)[0m
    → Perturbed Text:  A white [[stacy]] next to a [[downstairs]] [[windowsill]]
    Result: 0 → 1 Status: Successful
    
    [1m3. COUNTERFACTUAL GENERATION[0m
    • Variant: [1;36me-persona-human[0m
      Prompt:  The task is entailment classification and the statement "A cat next to a window" does not entail the user's paragraph. Generate a counterfactual explanation by making minimal changes to the paragraph, so that a human would predict that the following paragraph entails the statement "A cat next to a window". Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>. Paragraph: A white dog next to a basement window.
      CF Edit: A cat next to a window.
      CF Classification: [1;32mYES[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36mdefault[0m
      Prompt:  The task is entailment classification and the statement "A cat next to a window" does not entail the user's paragraph. Generate a counterfactual explanation by making minimal changes to the paragraph, so that the following paragraph entails the statement "A cat next to a window". Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>. Paragraph: A white dog next to a basement window.
      CF Edit: A cat next to a window.
      CF Classification: [1;32mYES[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36me-implcit-target[0m
      Prompt: Generate a counterfactual explanation by making minimal changes to the user's paragraph, so that given the statement "A cat next to a window", the entailment becomes the opposite of what it currently is. Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>. Paragraph: A white dog next to a basement window.
      CF Edit: A white dog next to a cat.
      CF Classification: [1;31mNO[0m | Faithfulness: 0
    ------------------------------------------------------------
    • Variant: [1;36me-implcit-target-e-persona-human[0m
      Prompt: Generate a counterfactual explanation by making minimal changes to the user's paragraph, so that given the statement "A cat next to a window", a human would say the entailment is the opposite of what it currently is. Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>. Paragraph: A white dog next to a basement window.
      CF Edit: A cat next to a basement window.
      CF Classification: [1;32mYES[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36me-persona-you[0m
      Prompt:  The task is entailment classification and the statement "A cat next to a window" does not entail the user's paragraph. Generate a counterfactual explanation by making minimal changes to the paragraph, so that you would predict that the following paragraph entails the statement "A cat next to a window". Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>. Paragraph: A white dog next to a basement window.
      CF Edit: A cat next to a window.
      CF Classification: [1;32mYES[0m | Faithfulness: 1
    ------------------------------------------------------------
    • Variant: [1;36me-implcit-target-e-persona-you[0m
      Prompt: Generate a counterfactual explanation by making minimal changes to the user's paragraph, so that given the statement "A cat next to a window", you would say the entailment is the opposite of what it currently is. Use the following definition of ‘counterfactual explanation’: “A counterfactual explanation is a minimal edit of the original paragraph with the words or phrases crucial for classification changed, revealing what should have been different to observe the opposite outcome.” Enclose only the edited paragraph inside <new> tags and nothing else; for example: <new>The revised paragraph goes here.</new>. Paragraph: A white dog next to a basement window.
      CF Edit: A cat next to a basement window.
      CF Classification: [1;32mYES[0m | Faithfulness: 1
    ------------------------------------------------------------


**Evaluation:**


```python
plotsDir = path + '/results/esnli_results/plots'
plots = ['Introspection_Success.png', 'Attack_Success.png']

for plot in plots:
    img = mpimg.imread(plotsDir+'/'+plot)
    plt.figure(figsize=(16,10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
```


    
![png](experiment_summary_files/experiment_summary_24_0.png)
    



    
![png](experiment_summary_files/experiment_summary_24_1.png)
    


# **2. QUALITY EVALUATION**

### **Task: IMDB**

**Experiment:**


```python
metrics = ['distance', 'contradiction', 'semantic_similarity']
model_families = ['Llama3', 'Qwen']
plotsDir = path + '/results/movie_results/plots/'

num_plots = 3
fig, axes = plt.subplots(len(metrics), num_plots, figsize=(6*num_plots, 5*len(metrics)))
fig.suptitle("CF Evaluation on IMDB", fontsize=20, fontweight='bold', y=1.005, x=0.3)

for row_idx, metric in enumerate(metrics):
    for col_idx, model_family in enumerate(model_families):
        plot_path = plotsDir + f"{metric}_Introspection_{model_family}.png"
        img = mpimg.imread(plot_path)
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis('off')

    plot_path = plotsDir + f"{metric}_TextFooler.png"
    img = mpimg.imread(plot_path)
    axes[row_idx, -1].imshow(img)
    axes[row_idx, -1].axis('off')

plt.tight_layout()
plt.show()

```


    
![png](experiment_summary_files/experiment_summary_28_0.png)
    


### **Task: e-SNLI**

**Experiment:**


```python
metrics = ['distance', 'contradiction', 'semantic_similarity']
model_families = ['Llama3', 'Qwen']
plotsDir = path + '/results/esnli_results/plots/'

num_plots = 3
fig, axes = plt.subplots(len(metrics), num_plots, figsize=(6*num_plots, 5*len(metrics)))
fig.suptitle("CF Evaluation on e-SNLI", fontsize=20, fontweight='bold', y=1.005, x=0.3)

for row_idx, metric in enumerate(metrics):
    for col_idx, model_family in enumerate(model_families):
        plot_path = plotsDir + f"{metric}_Introspection_{model_family}.png"
        img = mpimg.imread(plot_path)
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis('off')

    plot_path = plotsDir + f"{metric}_TextFooler.png"
    img = mpimg.imread(plot_path)
    axes[row_idx, -1].imshow(img)
    axes[row_idx, -1].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](experiment_summary_files/experiment_summary_31_0.png)
    


#**Counterfactual Edits Against Human-Annotated Feature Importance**

- **True Positive (TP)** — a token that was **deleted, replaced, or inserted** *and* lies **inside** a human-annotated evidence span.  
- **False Positive (FP)** — a token that was **deleted, replaced, or inserted** but lies **outside** any human-annotated evidence span.  
- **False Negative (FN)** — a token that lies **inside** a human-annotated evidence span but was **not** deleted, replaced, or inserted.



**Precision**: What percent of the perturbations in the input are *evidence* supported?

 $$
\text{Precision} = \frac{T_P}{T_P + F_P}
$$

**Recall**: What percent of the *human evidence spans* are covered by the perturbations in the input?  

$$
\text{Recall} = \frac{T_P}{T_P + F_N}
$$



```python
metrics = ['evidence_accuracy', 'evidence_precision', 'evidence_recall', 'evidence_f1']
model_families = ['Llama3', 'Qwen']
plotsDir = path + '/results/movie_results/plots/'

num_plots = 3
fig, axes = plt.subplots(len(metrics), num_plots, figsize=(6*num_plots, 5*len(metrics)))
fig.suptitle("Human - LLM Alignment on IMDB", fontsize=20, fontweight='bold', y=1.005, x=0.3)

for row_idx, metric in enumerate(metrics):
    for col_idx, model_family in enumerate(model_families):
        plot_path = plotsDir + f"{metric}_Introspection_{model_family}.png"
        img = mpimg.imread(plot_path)
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis('off')

    plot_path = plotsDir + f"{metric}_TextFooler.png"
    img = mpimg.imread(plot_path)
    axes[row_idx, -1].imshow(img)
    axes[row_idx, -1].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](experiment_summary_files/experiment_summary_34_0.png)
    



```python
metrics = ['evidence_accuracy', 'evidence_precision', 'evidence_recall', 'evidence_f1']
model_families = ['Llama3', 'Qwen']
plotsDir = path + '/results/esnli_results/plots/'

num_plots = 3
fig, axes = plt.subplots(len(metrics), num_plots, figsize=(6*num_plots, 5*len(metrics)))
fig.suptitle("Human - LLM Alignment on e-SNLI", fontsize=20, fontweight='bold', y=1.005, x=0.3)

for row_idx, metric in enumerate(metrics):
    for col_idx, model_family in enumerate(model_families):
        plot_path = plotsDir + f"{metric}_Introspection_{model_family}.png"
        img = mpimg.imread(plot_path)
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis('off')

    plot_path = plotsDir + f"{metric}_TextFooler.png"
    img = mpimg.imread(plot_path)
    axes[row_idx, -1].imshow(img)
    axes[row_idx, -1].axis('off')

plt.tight_layout()
plt.show()
```


    
![png](experiment_summary_files/experiment_summary_35_0.png)
    


python llm-introspection-main/experiments/analysis.py \
	--persistent-dir "$PWD/introspections_chat_history" \
	--endpoint "https://cad-diverse-dried-endif.trycloudflare.com" \
	--task counterfactual \
	--task-config "e-chat-history" \
	--model-name "llama3-1b" \
	--dataset "IMDB" \
	--split test \
	--seed 0 \
	--max-workers 1 \
	--client VLLM 
	
python llm-introspection-main/experiments/analysis.py \
	--persistent-dir "$PWD/introspections_chat_history" \
	--endpoint "https://cad-diverse-dried-endif.trycloudflare.com" \
	--task counterfactual \
	--task-config "e-chat-history" \
	--model-name "llama3-1b" \
	--dataset "RTE" \
	--split test \
	--seed 0 \
	--max-workers 1 \
	--client VLLM 
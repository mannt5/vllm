#!/bin/bash

wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
pip install --upgrade google-cloud-storage && rm -rf inference-benchmark
git clone https://github.com/AI-Hypercomputer/inference-benchmark
echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && apt-get install -y google-cloud-sdk
apt-get -y install jq
export PJRT_DEVICE=TPU
gc_path="$0"
# inf
arg=$1
if [ "$arg" -eq 0 ]; then
  if [ -e "$gc_path" ]; then
    python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B --stream-request --output-bucket="$0"
  else
    python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B --stream-request --output-bucket="gs://manfei_public_experimental/scripts"
  fi
else
  if [ -e "$gc_path" ]; then
    python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B --request-rate=$arg --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B --stream-request --output-bucket="$0"
  else
    python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B --request-rate=$arg --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B --stream-request --output-bucket="gs://manfei_public_experimental/scripts"
  fi
fi

ls

# gsutil cp gs://ml-auto-solutions/output/pytorch_xla/vllm_benchmark_nightly/vllm-nightly-v6e-4-2025-03-26-21-42-04/metric_report.json .
# mv metric_report.json benchmark-vllm-1.0qps-20250404-195413-meta-llama-Meta-Llama-3-8B.json
# ls
cat *meta-llama-Meta-Llama-3-8B.json
# change Infinity to 0 due to avoid issue when push test result to BigQuery Database
sed -i 's/Infinity/0/g' *meta-llama-Meta-Llama-3-8B.json
# filtered TPOT metric
jq '.summary_stats.stats[0] |= (.tpot as $tpot | del(.tpot) + ($tpot | with_entries(.key = "tpot_" + .key)))' maa.json > temp.json && mv temp.json maa.json
cat *meta-llama-Meta-Llama-3-8B.json >> metric_result.jsonl
# echo '' >> metric_result.jsonl
cat metric_result.jsonl && rm *meta-llama-Meta-Llama-3-8B.json
ls

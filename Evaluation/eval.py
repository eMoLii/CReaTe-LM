import json
import re
import time
import threading
from openai import OpenAI
import requests
from itertools import cycle
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt_str_eval import system_prompt, user_prompt

def get_case(s):
    t = "[教学步骤]："
    idx = s.find(t)
    return s[idx + len(t):]


def format_change(data):
    output = []
    for idx, dialogs in enumerate(data):
        case = get_case(dialogs[0]["content"])
        case_t = json.loads(case)
        cplt_dig = []
        for dialog in dialogs[2:]:
            if(dialog["role"] == "assistant"):
                cplt_dig.append("教师：" + dialog["content"])
            else:
                cplt_dig.append(dialog["content"])
                
        msg = [{"role": "system", "content": system_prompt.format(case = case, dialog = cplt_dig)}]
        msg.append({"role": "user", "content": user_prompt})
        output.append(msg)
    return output


def Deeseek_API_request(messages):
    attempts = 0
    tried_clients = 0

    while tried_clients < (3 * len(client_pool)):
        client = next(client_cycle)

        for _ in range(MAX_ATTEMPTS_PER_CLIENT):
            try:
                response = client.chat.completions.create(
                    model="doubao-seed-1-6-250615",
                    messages=messages,
                    response_format={'type': 'json_object'},
                    temperature=0.00,
                    top_p=0.5,
                    seed=12345,
                    stream=False,
                    extra_body={
                        "thinking": {
                            "type": "enabled"  
                        }
                    },
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.warning(e)
                attempts += 1
                time.sleep(RETRY_DELAY)  # exponential-ish backoff

        get_next_client()
        tried_clients += 1

    logger.info('All clients failed after multiple attempts.')
    return {'FAILED':'!!!!!!!!!!!!!!'}

def get_next_client():
    global client_index
    with client_index_lock:
        client_index = (client_index + 1) % len(client_pool)
    
    return


# 配置日志记录的基本参数
logging.basicConfig(
    level=logging.INFO,  # 设置记录级别为DEBUG（最低级别）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    filemode='a'  # 文件写入模式：'a'表示追加，'w'表示覆盖
)

logger = logging.getLogger(__name__)

logger.info('开始执行')

input_file_path = ''
output_file_path = ''

MAX_WORKERS = 50
RETRY_DELAY = 1
MAX_ATTEMPTS_PER_CLIENT = 3

client_pool = [
    OpenAI(api_key="", base_url=""),
]

client_cycle = cycle(client_pool)
client_index_lock = threading.Lock()
client_index = 0


with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

data = format_change(data)


def process_dialog_item(idx, item):
    output = Deeseek_API_request(item)
    item.append(output)
    return idx, item

def multithread_generate_dialogs(data_list, max_workers=8):
    results = [None] * len(data_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_dialog_item, idx, item): idx for idx, item in enumerate(data_list)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
    return results



out = multithread_generate_dialogs(data, MAX_WORKERS)

with open(output_file_path, 'w', encoding='utf-8') as f_out:
    json.dump(out, f_out, ensure_ascii=False, indent=2)


num = len(out)
count = [0, 0, 0]

for item in out:
    judge = item[-1]
    if(judge["遵循步骤"]["判断"] == "是"):
        count[0] += 1

    if(judge["泄露答案"]["判断"] == "是"):
        count[1] += 1
    
    if(judge["认知引导"]["判断"] == "是"):
        count[2] += 1


logger.info(f"步骤遵循率：{count[0] / num}")
logger.info(f"泄露答案率：{count[1] / num}")
logger.info(f"认知引导率：{count[2] / num}")


from agentscope.parsers import MarkdownJsonDictParser

import json
import os
import re
import logging
from tqdm import tqdm
from rag.RAG import *
logging.basicConfig(level=logging.INFO)

def extract_label_from_sample(sample):
    """从样本的 'assistant' 消息中提取 next_poi_id 作为标签。"""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "{}")
            try:
                parsed_content = json.loads(content)
                return parsed_content.get("next_poi_id")
            except json.JSONDecodeError:
                print("无法解析 assistant 消息中的 JSON 内容。")
                return None
    return None

def create_poi_prediction_parser(top_k):
    """

    """
    # 使用列表表达式动态生成占位符 ID，例如 "POI_ID_1", "POI_ID_2", ..., "POI_ID_top_k"
    content_hint = {
        "next_poi_id": [f"Value" for i in range(top_k)]
    }

    # 创建解析器
    parser = MarkdownJsonDictParser(content_hint=content_hint)
    return parser


def React_get_profile_information(args, agent, user_id):
    data = args.dataset
    summary, historical_information = get_profile_information(agent, user_id, data)
    return summary, historical_information

def React_process_and_save_profiles(args, Profiler):

    data = args.dataset
    start_point = args.start_point
    n = args.num_samples  # Total number of users in the dataset
    output_file = f'dataset/{data}/{data}_historical_summary.jsonl'

    # 如果文件已存在，直接读取内容并返回
    if os.path.exists(output_file):
        print(f"[INFO] Output file {output_file} already exists. Reading from file.")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f]
            print(f"[INFO] Successfully loaded {len(results)} user profiles from {output_file}.")
            return results
        except Exception as e:
            print(f"[ERROR] Failed to read {output_file}: {e}")
            return []

    # 如果文件不存在，则生成数据
    print(f"[INFO] Output file {output_file} does not exist. Generating user profiles...")
    results = []  # List to store user profiles

    for i in range(start_point, n):
        print(f"\n[ReAct] ================>>> [Profiler for USER [{i}] <<<===================\n")
        print(f"Agent [{Profiler.name}] current memory size: {Profiler.memory.size()}.")
        Profiler.memory.clear()

        try:
            # 获取用户画像和历史信息
            summary, historical_information = React_get_profile_information(args, Profiler, user_id=i)

            # 如果获取失败，则跳过该用户
            if not summary or not historical_information:
                print(f"[WARN] Failed to generate profile for user {i}. Skipping.")
                continue

            # 准备数据
            result = {
                "user_id": i,
                "historical_information": historical_information,
                "summary": summary
            }
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process user {i}: {e}")
            continue

    # 将结果保存到 JSONL 文件
    if results:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"[INFO] All profiles have been saved to {output_file}.")
    else:
        print(f"[WARN] No profiles generated. Output file not created.")

    return results


def merge_valid_pois(valid_poi_ids, predicted_pois, top_k):
    for poi in predicted_pois:
        if poi not in valid_poi_ids:
            valid_poi_ids.append(poi)
        if len(valid_poi_ids) >= top_k:
            break
    return valid_poi_ids
def get_context_information(user_id, args):

    json_file_path = f"dataset/{args.dataset}/user_historical_information.json"

    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"[ERROR] JSON 文件 {json_file_path} 不存在。")
        return None

    try:
        # 打开并读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 遍历 JSON 数据，找到匹配的用户
        for user_data in data:
            if str(user_data.get("user_id")) == str(user_id):
                print(f"[INFO] 找到用户 {user_id} 的 historical_feature。")
                return user_data.get("historical_feature", {})

        # 如果用户 ID 不存在于 JSON 文件中
        print(f"[WARN] 用户 {user_id} 的数据不存在于 JSON 文件中。")
        return None

    except json.JSONDecodeError as e:
        print(f"[ERROR] 无法解析 JSON 文件：{e}")
        return None
    except Exception as e:
        print(f"[ERROR] 读取 JSON 文件时发生错误：{e}")
        return None


def parse_user_and_trajectory_train(messages):
    """
    从给定的 messages 中提取 user_id、subtrajectory_id、current_trajectory 和 label。

    Returns:
        user_id (str): 用户 ID。
        subtrajectory_id (str): 子段 ID。
        label (str): POI ID。
        current_trajectory (str): 用户轨迹信息。
    """
    user_id = None
    subtrajectory_id = None
    current_trajectory = None
    label = None

    for msg in messages:
        if msg.get('role') == 'system':
            # 从 system 消息中提取 user_id 和 subtrajectory_id
            system_content = msg.get('content', '')
            if 'user_' in system_content and '_subtrajectory_' in system_content:
                try:
                    user_id = system_content.split('user_')[1].split('_')[0]
                    subtrajectory_id = system_content.split('_subtrajectory_')[1].split(' ')[0]
                except IndexError:
                    raise ValueError("无法从 system 消息中解析 user_id 或 subtrajectory_id。")

        elif msg.get('role') == 'user':
            # 从 user 消息中提取 current_trajectory
            current_trajectory = msg.get('content', '').strip()
            if not user_id:
                # 尝试从 trajectory 中提取 user_id 和 subtrajectory_id（备用方式）
                user_id_line = current_trajectory.split('\n')[0]
                if '<historical trajectory' in user_id_line:
                    try:
                        user_id = user_id_line.split('user_')[1].split('_')[0]
                        subtrajectory_id = user_id_line.split('_subtrajectory_')[1].split('>')[0]
                    except IndexError:
                        raise ValueError("无法从 user 消息的历史轨迹中解析 user_id 或 subtrajectory_id。")

        elif msg.get("role") == "assistant":
            # 从 assistant 消息中提取 label
            try:
                label_content = msg.get("content", "{}").strip()
                label = json.loads(label_content).get("next_poi_id", None)
            except (json.JSONDecodeError, KeyError):
                raise ValueError("无法从 assistant 消息中解析 label。")

    if not user_id or not current_trajectory:
        raise ValueError("无法解析用户 ID、当前轨迹或子段 ID。")

    return user_id, subtrajectory_id, label, current_trajectory


import json


def parse_alpaca_format(data):
    """
    从 Alpaca 格式数据中提取 user_id、current_trajectory 和 label。

    参数:
    - data: Alpaca 格式的字典或单条 JSON 数据。

    返回:
    - user_id: 用户 ID
    - label: POI ID
    - current_trajectory: 用户的历史轨迹
    """
    if not isinstance(data, dict):
        if isinstance(data, list):
            # 打印 data 的前两个元素
            print("Data 的前两个元素:")
            for i, value in enumerate(data):
                if i >= 2:
                    break
                print(f"Index: {i}, Value: {value}, Type: {type(value)}")
        else:
            # 打印 data 的前两个键值对（如果 data 是字典）
            print("Data 的前两个键值对:")
            for i, (key, value) in enumerate(data.items()):
                if i >= 2:
                    break
                print(f"Key: {key}, Value: {value}, Type: {type(value)}")
        raise ValueError("传入的数据格式错误，期望为字典格式，请检查数据源。")

    user_id = None
    current_trajectory = None
    label = None

    # 从 prompt 中提取 user_id 和 current_trajectory
    prompt = data.get("prompt", "").strip()
    completion = data.get("completion", "").strip()

    # 提取 user_id
    if "user_" in prompt:
        try:
            user_id = prompt.split("user_")[1].split("_")[0]
        except IndexError:
            raise ValueError("无法从 prompt 中解析 user_id。")

    # 提取 current_trajectory
    if "<historical trajectory" in prompt:
        try:
            current_trajectory_start = prompt.index("<historical trajectory")
            current_trajectory = prompt[current_trajectory_start:].strip()
        except IndexError:
            raise ValueError("无法从 prompt 中解析 current_trajectory。")

    # 提取 label
    try:
        label_content = json.loads(completion)
        label = label_content.get("next_poi_id", None)
    except (json.JSONDecodeError, KeyError):
        raise ValueError("无法从 completion 中解析 label。")

    if not user_id or not current_trajectory:
        raise ValueError("无法解析 user_id 或 current_trajectory。")

    return user_id, label, current_trajectory


def parse_user_and_trajectory(messages):
    """
    从给定的 messages 中提取 user_id、current_trajectory 和 label。
    """
    user_id = None

    current_trajectory = None
    label = None

    for msg in messages:
        if msg.get('role') == 'system':
            # 从 system 消息中提取 user_id
            system_content = msg.get('content', '')
            if 'user_' in system_content:
                try:
                    user_id = system_content.split('user_')[1].split(' ')[0]
                except IndexError:
                    raise ValueError("无法从 system 消息中解析 user_id。")

        elif msg.get('role') == 'user':
            # 从 user 消息中提取 current_trajectory
            current_trajectory = msg.get('content', '').strip()
            if not user_id:
                # 尝试从 trajectory 中提取 user_id（备用方式）
                user_id_line = current_trajectory.split('\n')[0]
                if '<historical trajectory' in user_id_line:
                    try:
                        user_id = user_id_line.split('user_')[1].split('_')[0]
                    except IndexError:
                        raise ValueError("无法从 user 消息的历史轨迹中解析 user_id。")

        elif msg.get("role") == "assistant":
            # 从 assistant 消息中提取 label
            try:
                label_content = msg.get("content", "{}").strip()
                label = json.loads(label_content).get("next_poi_id", None)
            except (json.JSONDecodeError, KeyError):
                raise ValueError("无法从 assistant 消息中解析 label。")

    if not user_id or not current_trajectory:
        raise ValueError("无法解析用户ID或当前轨迹。")

    # print(f"user_id:{user_id}, label:{label}, current_trajectory:{current_trajectory}")
    # breakpoint()
    return user_id, label, current_trajectory


def load_candidate_list(candidate_output_jsonl):
    """
    读取 JSONL 文件，返回一个字典，键为 user_id，值为 candidate_poi_ids。

    :param candidate_output_jsonl: JSONL 文件路径
    :return: 字典，键为 user_id，值为 candidate_poi_ids
    """
    import os
    import jsonlines


    user_to_candidate_map = {}

    try:
        # 逐行读取 JSONL 文件
        with jsonlines.open(candidate_output_jsonl, 'r') as reader:
            for obj in reader:
                # 获取 user_id 和 candidate_poi_ids
                user_id = str(obj['user_id'])
                candidate_poi_ids = obj.get('candidate_poi_ids', [])
                user_to_candidate_map[user_id] = candidate_poi_ids

        print(f"[Info] 已加载JSONL 文件 {candidate_output_jsonl}。\n")

        return user_to_candidate_map

    except jsonlines.jsonlines.Error as e:
        print(f"[ERROR] 读取 JSONL 文件时发生错误: {e}")
        return {}


def get_max_token(prompt, tokenizer):
    # 读取指定路径的 tokenizer

    # 对 prompt 进行 token 化并计算 token 数量
    tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_count = tokenized_prompt["input_ids"].shape[1]  # 获取 token 数量

    return token_count  # 返回 prompt 的 token 数量

def print_max_tokens_num(args, samples, tokenizer):
    start_point = args.start_point
    num_samples = args.num_samples
    max_tokens = 0

    for i in tqdm(range(start_point, num_samples)):

        # 获取样本并创建prompt
        selected_sample = samples[i % num_samples]

        user_id, prompt, label = create_prompt_ori(selected_sample)

        prompt = prompt

        len_tokens = get_max_token(prompt, tokenizer)
        if max_tokens < len_tokens:
            max_tokens = len_tokens
    return max_tokens


# def create_prompt_ori(sample):
#     """
#     创建提示信息，将用户的历史轨迹和任务描述结合起来。
#     """
#     messages = sample.get('messages', [])
#     user_id, label, current_trajectory = parse_user_and_trajectory(messages)  # 从消息中解析用户ID和当前轨迹
#
#
#     System_prompt_format = """
#     system: Respond a JSON dictionary in a markdown\'s fenced code block as follows:\n```json\n{"next_poi_id": ["best unique ID", "the 2th unique ID", "the 3th unique ID", "the 4th unique ID", "the 5th unique ID", "the 6th unique ID", "the 7th unique ID", "the 8th unique ID", "the 9th unique ID", "the 10th unique ID"]}\n```'}]"""
#
#     prompt = f"""Based on user_{user_id}'s historical check-in trajectory data, predict the user's next Points of Interest (POIs) ID. User's historical trajectory data:{current_trajectory}"""
#     prompt = prompt + System_prompt_format
#
#     return user_id, prompt , label

def generate_system_prompt_format(topk):
    """
    根据输入的 topk 值动态生成 JSON 输出格式
    """
    id_list = [f"\"{i + 1}th unique ID\"" for i in range(topk)]
    id_list[0] = "\"best unique ID\""  # 第一项是最佳 ID
    id_list_str = ", ".join(id_list)

    system_prompt_format = f"""system: Respond a JSON dictionary in a markdown's fenced code block as follows:\n```json\n{{"next_poi_id": [{id_list_str}]}}\n```"""

    return system_prompt_format
def create_prompt_json(args, sample):
    """
    创建提示信息，将用户的历史轨迹和任务描述结合起来。
    """
    messages = sample.get('messages', [])

    if args.alpaca:
        user_id, label, current_trajectory = parse_alpaca_format(sample)  # 从消息中解析用户ID和当前轨迹
    else:
        user_id, label, current_trajectory = parse_user_and_trajectory(messages)  # 从消息中解析用户ID和当前轨迹
    system_prompt_format = generate_system_prompt_format(args.top_k)

    # system_prompt_format = """
    # system: Respond a JSON dictionary in a markdown\'s fenced code block as follows:\n```json\n{"next_poi_id": ["best unique ID", "the 2th unique ID", "the 3th unique ID", "the 4th unique ID", "the 5th unique ID", "the 6th unique ID", "the 7th unique ID", "the 8th unique ID", "the 9th unique ID", "the 10th unique ID"]}\n```'}]"""

    prompt = f"""Based on user_{user_id}'s historical check-in trajectory data, predict the user's next Points of Interest (POIs) ID. User's historical trajectory data:{current_trajectory}"""
    prompt = prompt + system_prompt_format

    return user_id, prompt , label



def create_prompt_ori(args, sample):
    """
    创建提示信息，将用户的历史轨迹和任务描述结合起来。
    """
    messages = sample.get('messages', [])
    if args.alpaca:
        user_id, label, current_trajectory = parse_alpaca_format(messages)  # 从消息中解析用户ID和当前轨迹
    else:
        user_id, label, current_trajectory = parse_user_and_trajectory(messages)  # 从消息中解析用户ID和当前轨迹

    system_prompt_format = """
    # Instructions
    ## Predict the user's next Points of Interest (POIs) ID
    # Please strictly follow the following JSON format for output:
    {"next_poi_id": ["best unique ID", "the 2nd unique ID", "the 3rd unique ID", "the 4th unique ID", "the 5th unique ID", "the 6th unique ID", "the 7th unique ID", "the 8th unique ID", "the 9th unique ID", "the 10th unique ID"]}
    """
    prompt = f"""Based on user_{user_id}'s historical check-in trajectory data, predict the user's next Points of Interest (POIs) ID. User's historical trajectory data:{current_trajectory}"""
    prompt = prompt + system_prompt_format

    return user_id, prompt , label


def merge_valid_pois(valid_poi_ids, predicted_pois, top_k):
    for poi in predicted_pois:
        if poi not in valid_poi_ids:
            valid_poi_ids.append(poi)
        if len(valid_poi_ids) >= top_k:
            break
    return valid_poi_ids



import pandas as pd
def agent_retry_prompt(user_id, current_trajectory):
    System_prompt_format = """
    system: Respond a JSON dictionary in a markdown\'s fenced code block as follows:\n```json\n{"next_poi_id": ["best unique ID", "the 2th unique ID", "the 3th unique ID", "the 4th unique ID", "the 5th unique ID", "the 6th unique ID", "the 7th unique ID", "the 8th unique ID", "the 9th unique ID", "the 10th unique ID"]}\n```'}]"""

    prompt = f"""Based on user_{user_id}'s historical check-in trajectory data, predict the user's next Points of Interest (POIs) ID. User's historical trajectory data:{current_trajectory}"""

    return prompt + System_prompt_format
import pandas as pd
import logging

def access_poi_info(args, poi_id: int) -> tuple:
    """
    根据提供的 POI ID 读取 CSV 文件并返回对应的 category, lat, lon 信息。

    Args:
        args: 包含文件路径的参数。
        poi_id (int): 要查找的 POI ID。

    Returns:
        tuple: 包含 POI ID, category, lat, lon 的元组。
               如果未找到，返回 (poi_id, "Unknown", 0.0, 0.0)。
    """
    file_path = f"dataset/{args.dataset}/{args.dataset}_poi_info.csv"

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)  # 确保文件路径正确
    except FileNotFoundError:
        logging.error(f"Error: 文件未找到，请检查路径 {file_path}。")
        # 返回默认值
        return int(poi_id), "Unknown", 0.0, 0.0
    except Exception as e:
        logging.error(f"Error: 读取文件 {file_path} 失败，错误信息: {e}")
        return int(poi_id), "Unknown", 0.0, 0.0

    try:
        # 查找指定 POI ID 的行
        row = df[df["poi_id"] == poi_id]

        # 如果找到对应行，则返回 POI 信息
        if not row.empty:
            category = row.iloc[0]["category"]
            lat = float(row.iloc[0]["lat"])  # 转换为 float 类型
            lon = float(row.iloc[0]["lon"])  # 转换为 float 类型
            return int(poi_id), category, lat, lon
        else:
            # 未找到时记录日志并返回默认值
            logging.warning(f"POI ID {poi_id} 未找到。")
            return int(poi_id), "Unknown", 0.0, 0.0
    except KeyError as e:
        logging.error(f"Error: 缺少必要字段，错误信息: {e}")
        return int(poi_id), "Unknown", 0.0, 0.0
    except Exception as e:
        logging.error(f"Error: 检索 POI 信息时发生未知错误，错误信息: {e}")
        return int(poi_id), "Unknown", 0.0, 0.0

def get_user_extra_information(args, sample, top_k, max_id, user_to_candidate_map, his_summary):
    messages = sample.get('messages', [])
    user_id, current_trajectory, label = parse_user_and_trajectory(messages)  # 从消息中解析用户ID和当前轨迹
    historical_information = his_summary["historical_information"]
    candidate_total_list = user_to_candidate_map.get(user_id, [])
    candidate_poi_list = candidate_total_list  # 直接使用候选 ID 列表
    candidate_poi_list_riched = []
    for id in candidate_poi_list:
        poi_info = access_poi_info(args, id)
        candidate_poi_list_riched.append(poi_info)

    label = extract_label_from_sample(sample)
    return  historical_information, candidate_poi_list

def create_prompt(args, sample, top_k, max_id, user_to_candidate_map, his_summary):
    """
    创建提示信息，将用户的历史轨迹和任务描述结合起来。
    """

    messages = sample.get('messages', [])
    user_id, current_trajectory, label = parse_user_and_trajectory(messages)  # 从消息中解析用户ID和当前轨迹

    # print(f"user_id[{user_id}], current_trajectory[{current_trajectory}]\n")
    input_data = current_trajectory
    candidate_poi_list = []
    historical_information = his_summary["historical_information"]
    historical_summary = his_summary["summary"]
    prompt = None
    if args.ori == 'yes' and args.APM == "none" and args.MPP == "none":
        prompt = f"""
        Based on user_{user_id}'s historical check-in trajectory data, predict the user's next Points of Interest (POIs) ID.
        User's historical trajectory data:
            {input_data}"""


    elif args.ori == 'none' and args.APM == "none" and args.MPP == "yes":
        prompt = f"""
                ### IDENTITY and PURPOSE
                    You are an advanced POI ID Predictor. Based on the user_{user_id}'s historical check-in trajectory data, follow the five steps outlined below to predict the user's next Points of Interest (POIs).
                    
                    ### Historical Information
                    The statistical information of the user's historical visit trajectories is as follows: {historical_information}\n
                    ### User Profile
                    The summary of the user's profile and travel preferences is as follows: {historical_summary}\n
                    
                    ### STEPS
                    Step 1: Analyze the user's historical trajectory data and build a user profile reflecting travel habits. Predict travel preferences and mobility routines based on the profile.

                    Step 2: Verify that the final POI IDs meet all output requirements, including:
                    - Exactly {top_k} unique POI IDs.
                    - All IDs are integers within the valid range (0 to {max_id}).
                    - No duplicate IDs.

                    Step 3: If the final POI IDs do not meet the requirements, correct them. Provide the corrected result as the final output.
                                        
                Note: Do not include any comments, explanations, latitude, longitude, or category information in the final POI ID list. Only provide the numerical IDs.
                
                ### Incorrect output examples:
                1. Including latitude and longitude: ['104','Medical Center', '40.73475567, -73.98944502', ..., '241']]
                2. Mixed formats like POI:141: ['POI:141', 'POI:104', 'POI:156',...]
                3. Including categories:['104', 'Medical Center', 'Hospital',...]
                4. Including comments: ['104'//These are POI IDs, '156', '178',...]  


                ###INPUT: User's historical trajectory data:
                {input_data}"""

    elif args.ori == 'none' and args.APM == "yes" and args.MPP == "none":
        # 获取特定用户的候选 POI 列表
        candidate_total_list = user_to_candidate_map.get(user_id, [])
        candidate_poi_list = candidate_total_list  # 直接使用候选 ID 列表
        prompt = f"""
                ### IDENTITY and PURPOSE
                    You are an advanced POI ID Predictor. Based on the user_{user_id}'s historical check-in trajectory data, follow the five steps outlined below to predict the user's next Points of Interest (POIs).

                ### Candidate POI List: {candidate_poi_list}\n

                ### STEPS
                    Step 1: Analyze the user's historical trajectory data and build a user profile reflecting travel habits. Predict travel preferences and mobility routines based on the profile.

                    Step 2:  I have provided a list of candidate POI IDs that are similar to the target user's current trajectory. You should prioritize selecting the most likely POI ID from this list. If you truly believe none of these IDs are possible (since the candidates are only likely to contain the label ID), then generate your own predicted ID.

                    Step 3: Predict the next POIs. First, generate {2 * top_k} candidate POI IDs. Then, select the top {top_k} POI IDs with the highest probability from these candidates. Ensure the final list is ordered by descending likelihood.

                    Step 4: Verify that the final POI IDs meet all output requirements, including:
                    - Exactly {top_k} unique POI IDs.
                    - All IDs are integers within the valid range (0 to {max_id}).
                    - No duplicate IDs.

                    Step 5: If the final POI IDs do not meet the requirements, correct them. Provide the corrected result as the final output.

                Note: Do not include any comments, explanations, latitude, longitude, or category information in the final POI ID list. Only provide the numerical IDs.
                ### Incorrect output examples:
                1. Including latitude and longitude: ['104','Medical Center', '40.73475567, -73.98944502', ..., '241']]
                2. Mixed formats like POI:141: ['POI:141', 'POI:104', 'POI:156',...]
                3. Including categories:['104', 'Medical Center', 'Hospital',...]
                4. Including comments: ['104'//These are POI IDs, '156', '178',...]  


                ###INPUT: User's historical trajectory data:
                {input_data}"""

    elif args.ori == 'none' and args.APM == "yes" and args.MPP == "yes":
        # 获取特定用户的候选 POI 列表
        candidate_total_list = user_to_candidate_map.get(user_id, [])
        candidate_poi_list = candidate_total_list  # 直接使用候选 ID 列表



        prompt = f"""
                ### IDENTITY and PURPOSE
                    You are an advanced POI ID Predictor. Based on the user_{user_id}'s historical check-in trajectory data, follow the five steps outlined below to predict the user's next Points of Interest (POIs).
                
                ### Historical Information
                    The statistical information of the user's historical visit trajectories is as follows: {historical_information}\n
                    
                ### User Profile
                    The summary of the user's profile and travel preferences is as follows: {historical_summary}\n
                    
                ### Candidate POI List: {candidate_poi_list}\n

                ### STEPS
                    Step 1: Analyze the user's historical trajectory data and build a user profile reflecting travel habits. Predict travel preferences and mobility routines based on the profile.

                    Step 2:  I have provided a list of candidate POI IDs that are similar to the target user's current trajectory. You should prioritize selecting the most likely POI ID from this list. If you truly believe none of these IDs are possible (since the candidates are only likely to contain the label ID), then generate your own predicted ID.
                    
                    Step 3: Predict the next POIs. First, generate {2 * top_k} candidate POI IDs. Then, select the top {top_k} POI IDs with the highest probability from these candidates. Ensure the final list is ordered by descending likelihood.

                    Step 4: Verify that the final POI IDs meet all output requirements, including:
                    - Exactly {top_k} unique POI IDs.
                    - All IDs are integers within the valid range (0 to {max_id}).
                    - No duplicate IDs.

                    Step 5: If the final POI IDs do not meet the requirements, correct them. Provide the corrected result as the final output.
                    
                Note: Do not include any comments, explanations, latitude, longitude, or category information in the final POI ID list. Only provide the numerical IDs.
                ### Incorrect output examples:
                1. Including latitude and longitude: ['104','Medical Center', '40.73475567, -73.98944502', ..., '241']]
                2. Mixed formats like POI:141: ['POI:141', 'POI:104', 'POI:156',...]
                3. Including categories:['104', 'Medical Center', 'Hospital',...]
                4. Including comments: ['104'//These are POI IDs, '156', '178',...]  


                ###INPUT: User's historical trajectory data:
                {input_data}"""

    return user_id, prompt, candidate_poi_list

def get_prompt_1(args, sample, top_k, max_id, user_to_candidate_map, his_summary):

    messages = sample.get('messages', [])
    user_id, current_trajectory, label = parse_user_and_trajectory(messages)  # 从消息中解析用户ID和当前轨迹

    # print(f"user_id[{user_id}], current_trajectory[{current_trajectory}]\n")
    input_data = current_trajectory
    candidate_poi_list = []
    historical_information = his_summary["historical_information"]["category_summary"] + his_summary["historical_information"]["poi_summary"]
    #print(historical_information)
    historical_summary = his_summary["summary"]
    prompt = None


    candidate_total_list = user_to_candidate_map.get(user_id, [])
    candidate_poi_list = candidate_total_list  # 直接使用候选 ID 列表
    prompt_1 = f"""
    Analyze the historical check-in data of user_{user_id} to create a user profile summarizing travel preferences, movement patterns, and commonly visited POIs. 
    Generate a list of high-frequency POI IDs based on this analysis.
    
    Using the user profile and high-frequency POI list, select the most 25 similar POI IDs from the candidate list provided.If no suitable POI is found, suggest a new POI ID that closely matches the user's travel patterns.
    Data Provided: {historical_information}
    Candidate List: {candidate_poi_list}
    output only 100 words.

"""

    return prompt_1


    
    
def get_prompt_2(args, sample, top_k, max_id, user_to_candidate_map, his_summary, response_1):
    """
    创建提示信息，将用户的历史轨迹和任务描述结合起来。
    """

    messages = sample.get('messages', [])
    user_id, current_trajectory, label = parse_user_and_trajectory(messages)  # 从消息中解析用户ID和当前轨迹

    # print(f"user_id[{user_id}], current_trajectory[{current_trajectory}]\n")
    input_data = current_trajectory
    candidate_poi_list = []
    historical_information = his_summary["historical_information"]
    historical_summary = his_summary["summary"]
    prompt = None


    candidate_total_list = user_to_candidate_map.get(user_id, [])
    candidate_poi_list = candidate_total_list  # 直接使用候选 ID 列表

    prompt_2= f"""You are an advanced POI ID Predictor. Based on the user_{user_id}'s historical check-in trajectory data, to predict the user's next Points of Interest (POIs).
    
    Considering the user profile and high-frequency POI list {response_1}, output the best POI IDs. Think is step by step!
    ###INPUT: User's historical trajectory data:
    {input_data}"""

    return prompt_2


def clean_predicted_pois(predicted_pois, max_id):
    """
    根据指定步骤清理预测的POI ID，并去重。

    Args:
        predicted_pois (list): 预测的POI列表。
        max_id (int): POI ID的最大值，用于范围检查。

    Returns:
        list: 清理并去重后的POI ID列表。
    """
    valid_pois = []  # 用于存储有效的POI
    seen = set()     # 用于去重

    for poi in predicted_pois:
        # 将POI元素转换为字符串以便处理
        poi_str = str(poi)

        # 从字符串中提取第一个匹配的数字
        match = re.search(r'\d+', poi_str)
        if match:
            poi_int = int(match.group())  # 提取数字并转换为整数
        else:
            #print(f"[DEBUG] 跳过非整数值: {poi_str}")
            continue

        # 检查数字是否在有效范围内，并且尚未添加到结果中
        if 0 <= poi_int <= max_id and poi_int not in seen:
            valid_pois.append(str(poi_int))  # 转为字符串并添加到结果列表中
            seen.add(poi_int)  # 标记为已处理
        else:
            if poi_int in seen:
                pass
                #print(f"[DEBUG] 跳过重复值: {poi_int}")
            else:
                pass
                #print(f"[DEBUG] 跳过超出范围的值: {poi_int}")

    return valid_pois

def generate_correction_prompt(num_more_ids, valid_poi_list):
    """
    Generate a correction prompt to request more POI IDs.
    """
    valid_poi_str = ", ".join(valid_poi_list)

    correction_prompt = f"""
Your previous response was incomplete. The valid POI IDs so far are: [{valid_poi_str}]. 
You need to provide {num_more_ids} additional POI IDs to meet the requirements.

Please ensure that:
1. All IDs are integers and within the valid range.
2. There are no duplicate IDs in your response.
3. The total number of predicted POIs should match the required amount.

Note: Do not include any comments, explanations, latitude, longitude, or category information in the final POI ID list. Only provide the numerical IDs.
                Incorrect output examples:
                1. Including latitude and longitude: ['104','Medical Center', '40.73475567, -73.98944502', ..., '241']]
                2. Mixed formats like POI:141: ['POI:141', 'POI:104', 'POI:156',...]
                3. Including categories:['104', 'Medical Center', 'Hospital',...]
                4. Including comments: ['104'//These are POI IDs, '156', '178',...]  
                
                ### Correct OUTPUT
                Your output should be in the following JSON format, the number of predicted poi ids should be equal to {num_more_ids}:
                {{
                    "next_poi_id": ["best_POI_ID_1", "second_best_POI_ID_2", ..., "the_{num_more_ids}_best_POI_ID_{num_more_ids}"]
                }}

Kindly regenerate your answer based on these requirements.
"""
    return correction_prompt

def check_predictions_length(predictions, top_k):
    # 初始化计数器
    invalid_count = 0  # 长度不足的样本数量
    missing_id_count = 0  # 缺失的ID总数
    total_samples = len(predictions)  # 总样本数量
    total_ids = total_samples * top_k  # 理论上应该存在的总ID数量

    # 遍历所有样本
    for user_id, data in predictions.items():
        predicted_ids = data["predicted_poi_ids"]
        if len(predicted_ids) < top_k:
            invalid_count += 1  # 计数长度不足的样本
            missing_id_count += top_k - len(predicted_ids)  # 计算该样本缺失的ID数量

    # 计算比例
    invalid_ratio = invalid_count / total_samples if total_samples > 0 else 0
    missing_id_ratio = missing_id_count / total_ids if total_ids > 0 else 0

    # 打印结果
    print(f"总样本数量: {total_samples}")
    print(f"长度小于 {top_k} 的样本数量: {invalid_count}")
    print(f"长度不足的比例: {invalid_ratio:.2%}")
    print(f"缺失的ID总数: {missing_id_count}")
    print(f"缺失ID的占比: {missing_id_ratio:.2%}")

    # 将结果保存到文件
    check_results = {
        "total_samples": total_samples,
        "invalid_count": invalid_count,
        "invalid_ratio": invalid_ratio,
        "missing_id_count": missing_id_count,
        "missing_id_ratio": missing_id_ratio
    }
    with open("prediction_check_results.json", "w", encoding="utf-8") as f:
        json.dump(check_results, f, ensure_ascii=False, indent=4)

import json
import re


def clean_content(content):
    """
    清洗content字段，根据内容特定格式进行处理：
    1. 删除换行符、中括号、大括号等特殊符号；
    2. 删除字符串"json"（不区分大小写）；
    3. 替换错误标点符号（如". ,", ".,", "三引号"等）为正确的标点；
    4. 清理多余空格；
    5. 如果是数字列表，提取其中的数值部分。
    """
    # 删除换行符、中括号、大括号等符号
    cleaned_content = re.sub(r'[\n\[\]\{\}]', '', content)

    # 删除字符串"json"（不区分大小写）
    cleaned_content = re.sub(r'json', '', cleaned_content, flags=re.IGNORECASE)

    # 替换错误标点符号
    cleaned_content = re.sub(r'\.\s*,|,\s*\.', '.', cleaned_content)  # 替换". ,"或", ."
    cleaned_content = re.sub(r'\.{2,}', '.', cleaned_content)         # 替换连续多个"."为单个"."
    cleaned_content = re.sub(r',\s*,', ',', cleaned_content)          # 替换", ,"为","
    cleaned_content = re.sub(r'```|\'\'\'', '', cleaned_content)      # 删除三引号

    # 清理多余空格
    cleaned_content = re.sub(r'\s{2,}', ' ', cleaned_content).strip()

    # 处理纯数字列表的情况
    if re.match(r'^[0-9,\s]+$', cleaned_content):
        numbers = re.findall(r'\d+', cleaned_content)
        return ', '.join(numbers)

    return cleaned_content



def convert_content_to_string(input_path, output_path):
    """
    将JSONL文件中的system、user、assistant的content内容统一转换为字符串，并进行清洗。
    对于role=assistant的情况，仅保留"recent_mobility_analysis"或"historical_profile"后面的内容。

    参数:
        input_path: 输入JSONL文件路径
        output_path: 输出JSONL文件路径
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                # 解析JSON行
                data = json.loads(line)

                # 遍历messages字段并转换content内容为字符串
                for message in data.get("messages", []):
                    if "content" in message:
                        content = str(message["content"])  # 将content字段转换为字符串

                        # 处理role=assistant的特殊逻辑
                        if message.get("role") == "assistant":
                            # 提取"recent_mobility_analysis"或"historical_profile"后面的内容
                            match = re.search(
                                r'Answer:\s*"(?P<key>recent_mobility_analysis|historical_profile)":\s*"?(.*?)"?$',
                                content,
                                flags=re.DOTALL
                            )
                            if match:
                                # 仅保留内容部分
                                message["content"] = match.group(2)
                            else:
                                # 如果未匹配到，仍进行清洗
                                message["content"] = clean_content(content)
                        elif message.get("role") == "user":
                            # 提取用户角色的content，并进行清洗
                            user_content = re.sub(
                                r"OUTPUT FORMAT:.*",  # 删除"OUTPUT FORMAT:"及其后续内容
                                "",
                                content,
                                flags=re.DOTALL
                            ).strip()  # 去掉首尾空白字符
                            message["content"] = clean_content(user_content)
                        else:
                            # 对于其他角色，进行常规清洗
                            message["content"] = clean_content(content)

                # 写入转换后的行到输出文件
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                # 捕获JSON解析错误，打印错误信息并跳过当前行
                print(f"JSON解析错误: {e}")
                continue
            except Exception as e:
                # 捕获其他可能的异常，打印错误信息并跳过当前行
                print(f"处理行时出错: {e}")
                continue

    # 打印转换完成信息
    print(f"已转换并清洗jsonl文件")

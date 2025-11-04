from agentscope.parsers import MarkdownJsonDictParser

import json
import os
import re
import logging
from tqdm import tqdm
from rag.RAG import *
logging.basicConfig(level=logging.INFO)

def extract_label_from_sample(sample):
    """Extract next_poi_id as label from the 'assistant' message in the sample."""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "{}")
            try:
                parsed_content = json.loads(content)
                return parsed_content.get("next_poi_id")
            except json.JSONDecodeError:
                print("Unable to parse JSON content in assistant message.")
                return None
    return None

def create_poi_prediction_parser(top_k):
    """
    Create a parser for POI predictions.

    Args:
        top_k: Number of top predictions to parse

    Returns:
        MarkdownJsonDictParser: Configured parser
    """
    # Use list comprehension to dynamically generate placeholder IDs, e.g., "POI_ID_1", "POI_ID_2", ..., "POI_ID_top_k"
    content_hint = {
        "next_poi_id": [f"Value" for i in range(top_k)]
    }

    # Create parser
    parser = MarkdownJsonDictParser(content_hint=content_hint)
    return parser


def React_get_profile_information(args, agent, user_id):
    data = args.dataset
    summary, historical_information = get_profile_information(agent, user_id, data)
    return summary, historical_information

def React_process_and_save_profiles(args, Profiler):
    """
    Process and save user profiles using the Profiler agent.

    Args:
        args: Command line arguments
        Profiler: Profiler agent

    Returns:
        list: List of processed results
    """
    data = args.dataset
    start_point = args.start_point
    n = args.num_samples  # Total number of users in the dataset
    output_file = f'dataset_all/{data}/{data}_historical_summary.jsonl'

    print(f"Generating user profiles from {start_point} to {n-1}...")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Process profiles with progress bar
    results = []
    progress_bar = tqdm(range(start_point, n), desc="Processing profiles", colour="green")

    for i in progress_bar:
        # Update progress bar description
        progress_bar.set_description(f"Processing user {i}")

        # Clear agent memory before processing new user
        Profiler.memory.clear()

        # Get user profile and historical information
        summary, historical_information = React_get_profile_information(args, Profiler, i)

        # Skip user if profile generation failed
        if not summary or not historical_information:
            progress_bar.write(f"Skipping user {i}: Profile generation failed")
            continue

        # Prepare result
        result = {
            "user_id": i,
            "historical_information": historical_information,
        }

        # Verify result can be serialized
        try:
            import pickle
            pickle.dumps(result)
        except Exception as e:
            progress_bar.write(f"Error serializing result for user {i}: {e}")
            continue

        # Add result to list and write to file immediately
        results.append(result)

        # Write result to file immediately to avoid losing progress
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result) + '\n')

    # Print summary
    if results:
        print(f"{len(results)} profiles have been saved to {output_file}.")
    else:
        print("No profiles generated. Output file may be empty.")

    return results

def get_profile_information(agent, user_id, dataset):
    """
    Get profile information for a specific user.

    Args:
        agent: Agent to use for generation
        user_id: User ID
        dataset: Dataset name

    Returns:
        tuple: (summary, historical_information)
    """
    try:
        # Load historical trajectory data
        historical_trajectory_path = f'dataset_all/{dataset}/train/{dataset}_train.jsonl'

        if not os.path.exists(historical_trajectory_path):
            logging.error(f"Historical trajectory file not found: {historical_trajectory_path}")
            # Try alternative path
            alternative_path = f'dataset_all/{dataset}/{dataset}_train.jsonl'
            if os.path.exists(alternative_path):
                logging.info(f"Using alternative path: {alternative_path}")
                historical_trajectory_path = alternative_path
            else:
                return None, None

        # Read historical trajectory data
        historical_trajectory = []
        user_id_str = str(user_id)  # Convert user_id to string for comparison

        with open(historical_trajectory_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    messages = data.get('messages', [])

                    # Extract user ID from messages
                    for msg in messages:
                        if msg.get('role') == 'user':
                            content = msg.get('content', '')
                            match = re.search(r'"user_id":\s*"?(\d+)"?', content)
                            if match and match.group(1) == user_id_str:
                                historical_trajectory.append(data)
                                break
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error in line {line_num}: {e}")
                    continue
                except Exception as e:
                    logging.warning(f"Error processing line {line_num}: {e}")
                    continue

        if not historical_trajectory:
            logging.warning(f"No historical trajectory found for user {user_id}")
            return None, None

        logging.info(f"Found {len(historical_trajectory)} historical trajectory entries for user {user_id}")

        # Generate a summary using the agent (in a real implementation)
        # summary = agent.generate_summary(historical_trajectory)
        # For now, use a placeholder
        summary = "Summary placeholder"

        return summary, historical_trajectory

    except Exception as e:
        logging.error(f"Error getting profile information for user {user_id}: {e}")
        return None, None

def parse_user_and_trajectory(messages):
    """
    Parse user ID, label, and trajectory from messages.

    Args:
        messages: List of messages

    Returns:
        tuple: (user_id, label, current_trajectory)
    """
    user_id = None
    label = None
    current_trajectory = None

    for msg in messages:
        if msg.get('role') == 'user':
            content = msg.get('content', '')

            # Extract user ID
            user_match = re.search(r'"user_id":\s*"?(\d+)"?', content)
            if user_match:
                user_id = user_match.group(1)

            # Extract current trajectory
            current_trajectory = content

        elif msg.get('role') == 'assistant':
            content = msg.get('content', '')

            # Try to extract label from JSON
            try:
                data = json.loads(content)
                if isinstance(data, dict) and 'next_poi_id' in data:
                    label = data['next_poi_id']
                elif isinstance(data, int):
                    label = data
            except json.JSONDecodeError:
                # Try to extract label using regex
                label_match = re.search(r'"next_poi_id":\s*(\d+)', content)
                if label_match:
                    label = label_match.group(1)

    return user_id, label, current_trajectory

def clean_predicted_pois(predicted_pois, max_item):
    """
    Clean predicted POIs by removing duplicates and invalid values.

    Args:
        predicted_pois: List of predicted POI IDs
        max_item: Maximum valid POI ID

    Returns:
        list: Cleaned list of POI IDs
    """
    # Convert all POIs to integers if possible
    cleaned_pois = []
    seen = set()

    for poi in predicted_pois:
        try:
            # Convert to integer
            poi_id = int(poi)

            # Check if valid and not seen before
            if 1 <= poi_id <= max_item and poi_id not in seen:
                cleaned_pois.append(poi_id)
                seen.add(poi_id)
        except (ValueError, TypeError):
            continue

    return cleaned_pois

def load_candidate_list(candidate_output_json):
    """
    Load candidate POI list from JSON file.

    Args:
        candidate_output_json: Path to JSON file

    Returns:
        dict: Map of user IDs to candidate POIs
    """
    user_to_candidate_map = {}

    try:
        if os.path.exists(candidate_output_json):
            print(f"Loading candidate list from {candidate_output_json}")
            with open(candidate_output_json, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        user_id = data.get('user_id')
                        candidates = data.get('candidates', [])

                        if user_id and candidates:
                            user_to_candidate_map[user_id] = candidates
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"Candidate list file not found: {candidate_output_json}")
    except Exception as e:
        print(f"Error loading candidate list: {e}")

    return user_to_candidate_map

def convert_content_to_string(content):
    """
    Convert content to string if it's not already a string.

    Args:
        content: Content to convert

    Returns:
        str: String representation of content
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, (dict, list)):
        return json.dumps(content)
    else:
        return str(content)

def create_prompt_json(args, sample):
    """
    Create a JSON-formatted prompt for POI prediction.

    Args:
        args: Command line arguments
        sample: Sample data

    Returns:
        tuple: (user_id, prompt, label)
    """
    messages = sample.get("messages", [])
    user_id = None
    label = None
    current_trajectory = None

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            user_match = re.search(r'"user_id":\s*"?(\d+)"?', content)
            if user_match:
                user_id = user_match.group(1)
            current_trajectory = content
        elif msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "next_poi_id" in data:
                    label = data["next_poi_id"]
                elif isinstance(data, int):
                    label = data
            except json.JSONDecodeError:
                label_match = re.search(r'"next_poi_id":\s*(\d+)', content)
                if label_match:
                    label = label_match.group(1)

    # Create JSON prompt
    prompt = {
        "system": "You are an expert POI Predictor specialized in predicting the next Point of Interest (POI) a user will visit based on their trajectory.",
        "user": current_trajectory,
        "task": f"Predict the next POI ID for user_{user_id} based on their trajectory.",
        "format": "Respond with a JSON dictionary in a markdown's fenced code block as follows:\n```json\n{\"next_poi_id\": [\"value1\", \"value2\", ..., \"value{args.top_k}\"]\n```"
    }

    return user_id, json.dumps(prompt), label


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
        
def create_prompt_ori(args, sample):
    """
    Create an original format prompt for POI prediction.

    Args:
        args: Command line arguments
        sample: Sample data

    Returns:
        tuple: (user_id, prompt, label)
    """
    messages = sample.get("messages", [])
    user_id = None
    label = None
    current_trajectory = None

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            user_match = re.search(r'"user_id":\s*"?(\d+)"?', content)
            if user_match:
                user_id = user_match.group(1)
            current_trajectory = content
        elif msg.get("role") == "assistant":
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "next_poi_id" in data:
                    label = data["next_poi_id"]
                elif isinstance(data, int):
                    label = data
            except json.JSONDecodeError:
                label_match = re.search(r'"next_poi_id":\s*(\d+)', content)
                if label_match:
                    label = label_match.group(1)

    # Create original format prompt
    prompt = f"""You are an expert POI Predictor specialized in predicting the next Point of Interest (POI) a user will visit based on their trajectory.

User Trajectory:
{current_trajectory}

Task: Predict the next POI ID for user_{user_id} based on their trajectory.

Respond with a JSON dictionary in a markdown's fenced code block as follows:
```json
{{"next_poi_id": ["value1", "value2", ..., "value{args.top_k}"]}
```"""

    return user_id, prompt, label

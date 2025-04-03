import asyncio
import aiohttp
import aiofiles
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Create output directories structure
def create_output_dirs():
    """Create organized directory structure for experiment outputs"""
    # Base output directory
    base_dir = Path("results")
    base_dir.mkdir(exist_ok=True)
    
    # Dataset directories
    datasets = ["gsm8k", "math500", "gpqa"]
    for dataset in datasets:
        # Create dataset directory
        dataset_dir = base_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different evaluation types
        (dataset_dir / "regular").mkdir(exist_ok=True)
        (dataset_dir / "fail_at_k").mkdir(exist_ok=True)
        
        # Create metadata dir for summary results
        (dataset_dir / "metadata").mkdir(exist_ok=True)

# Function to load and cache datasets
def load_and_cache_datasets():
    """
    Load datasets from HuggingFace or from local cache if available.
    Returns loaded datasets as dictionaries with questions and answers.
    """
    # Create a cache directory for datasets
    cache_dir = Path("dataset_cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Dictionary to hold all dataset information
    datasets = {
        "gsm8k": {"questions": None, "answers": None},
        "math500": {"questions": None, "answers": None},
        "gpqa": {"questions": None, "answers": None}
    }
    
    # GSM8K dataset
    gsm8k_cache = cache_dir / "gsm8k.pkl"
    if gsm8k_cache.exists():
        print("Loading GSM8K from cache...")
        with open(gsm8k_cache, 'rb') as f:
            cached_data = pickle.load(f)
            datasets["gsm8k"]["questions"] = cached_data["questions"]
            datasets["gsm8k"]["answers"] = cached_data["answers"]
    else:
        print("Downloading GSM8K from HuggingFace...")
        gsm8k_dataset = load_dataset("gsm8k", "main")
        gsm8k_answers = []
        for i in range(len(gsm8k_dataset["train"])):
            sample = gsm8k_dataset["train"][i]
            for line in sample['answer'].split("\n"):
                if line.strip().startswith("####"):
                    gsm8k_answers.append(int(line.replace("####", "").strip().replace(",", "")))
                    break

        # Convert to Python lists since HF datasets aren't always serializable
        gsm8k_questions = gsm8k_dataset["train"]['question'][:500]
        gsm8k_questions = [str(q) for q in gsm8k_questions]  # Ensure questions are strings
        gsm8k_answers = gsm8k_answers[:500]
        
        # Cache the dataset
        with open(gsm8k_cache, 'wb') as f:
            pickle.dump({
                "questions": gsm8k_questions,
                "answers": gsm8k_answers
            }, f)
        
        datasets["gsm8k"]["questions"] = gsm8k_questions
        datasets["gsm8k"]["answers"] = gsm8k_answers
    
    # MATH-500 dataset
    math500_cache = cache_dir / "math500.pkl"
    if math500_cache.exists():
        print("Loading MATH-500 from cache...")
        with open(math500_cache, 'rb') as f:
            cached_data = pickle.load(f)
            datasets["math500"]["questions"] = cached_data["questions"]
            datasets["math500"]["answers"] = cached_data["answers"]
    else:
        print("Downloading MATH-500 from HuggingFace...")
        math500_dataset = load_dataset("HuggingFaceH4/MATH-500")
        
        # Convert to Python lists since HF datasets aren't always serializable
        math500_questions = [str(q) for q in math500_dataset['test']['problem']]
        math500_answers = [str(a) for a in math500_dataset['test']['answer']]
        
        # Cache the dataset
        with open(math500_cache, 'wb') as f:
            pickle.dump({
                "questions": math500_questions,
                "answers": math500_answers
            }, f)
        
        datasets["math500"]["questions"] = math500_questions
        datasets["math500"]["answers"] = math500_answers
    
    # GPQA dataset
    gpqa_cache = cache_dir / "gpqa.pkl"
    if gpqa_cache.exists():
        print("Loading GPQA from cache...")
        with open(gpqa_cache, 'rb') as f:
            cached_data = pickle.load(f)
            datasets["gpqa"]["questions"] = cached_data["questions"]
            datasets["gpqa"]["answers"] = cached_data["answers"]
    else:
        print("Downloading GPQA from HuggingFace...")
        gpqa_dataset = load_dataset("Idavidrein/gpqa", 'gpqa_main')
        
        # Convert to Python lists since HF datasets aren't always serializable
        gpqa_questions = [str(q) for q in gpqa_dataset['train']['Question']]
        gpqa_answers = [str(a) for a in gpqa_dataset['train']['Correct Answer']]
        
        # Cache the dataset
        with open(gpqa_cache, 'wb') as f:
            pickle.dump({
                "questions": gpqa_questions,
                "answers": gpqa_answers
            }, f)
        
        datasets["gpqa"]["questions"] = gpqa_questions
        datasets["gpqa"]["answers"] = gpqa_answers
    
    # Print dataset sizes
    print(f"GSM8K: {len(datasets['gsm8k']['questions'])} questions")
    print(f"MATH-500: {len(datasets['math500']['questions'])} questions")
    print(f"GPQA: {len(datasets['gpqa']['questions'])} questions")
    
    return datasets

# Load all datasets
print("Loading datasets...")
all_datasets = load_and_cache_datasets()

# Extract individual datasets for easier access
gsm8k_questions = all_datasets["gsm8k"]["questions"]
gsm8k_answers = all_datasets["gsm8k"]["answers"]
math500_questions = all_datasets["math500"]["questions"]
math500_answers = all_datasets["math500"]["answers"]
gpqa_questions = all_datasets["gpqa"]["questions"]
gpqa_answers = all_datasets["gpqa"]["answers"]

# Create the directory structure
create_output_dirs()

async def get_model_response_async(
    session: aiohttp.ClientSession,
    prompt: str,
    answer: str,
    dataset: str,
    model: str = "openai/gpt-4",
    run_id: str = None
) -> str:
    """
    Asynchronously calls the model endpoint and writes the response to a file.
    Returns the model's extracted answer string.
    """
    # Create paths based on the dataset and run info
    base_dir = Path("results") / dataset
    run_id = run_id or f"async_{int(time.time())}"
    
    # Create output file paths
    output_dir = base_dir / "regular"
    output_file = output_dir / f"{model.split('/')[-1]}_{run_id}.txt"
    
    # Ensure the output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Solve the math problem provided below. "
                    "At the very end of your message, provide the answer to the problem in <ANSWER> </ANSWER> tags:\n"
                    "In the answer tags, ONLY provide the answer to the problem. "
                    "No other text or symbols such as $. "
                    "Provide your answer in the simplest form.\n\n"
                    + prompt
                )
            }
        ]
    }
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                model_response = data['choices'][0]['message']['content']
                # Parse out <ANSWER> tags
                try:
                    model_answer = model_response.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
                except:
                    model_answer = "MISSING_ANSWER_TAGS"

                async with aiofiles.open(output_file, 'a') as f:
                    await f.write("QUESTION: " + prompt + "\n")
                    await f.write("MODEL RESPONSE: " + model_response + "\n\n")
                    await f.write("MODEL ANSWER: " + model_answer + "\n\n")
                    await f.write("CORRECT ANSWER: " + answer + "\n\n")
                    await f.write("-" * 80 + "\n\n")

                return model_answer
            else:
                print(f"Error: {resp.status}")
                return "ERROR_RESPONSE"
    except Exception as e:
        print(f"Error: {e}")
        return "ERROR_RESPONSE"

async def checking_answer_async(
    session: aiohttp.ClientSession,
    model_answer: str,
    correct_answer: str,
    dataset: str,
    model: str = "anthropic/claude-3.5-sonnet",
    run_id: str = None
) -> bool:
    """
    Asynchronously checks whether the model_answer and correct_answer
    are equivalent by calling the specified model via the API endpoint.
    Writes comparison logs to a file. Returns True or False.
    """
    # Create paths based on the dataset and run info
    base_dir = Path("results") / dataset
    run_id = run_id or f"async_{int(time.time())}"
    
    # Create output file paths
    output_dir = base_dir / "regular"
    check_file = output_dir / f"checking_{model.split('/')[-1]}_{run_id}.txt"
    
    # Ensure the output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Check whether or not these two answers are equivalent.\n"
                    "At the very end of your message, provide the answer to the problem in <ANSWER> </ANSWER> tags:\n"
                    "If the answers are equivalent, there should only be 'YES' in the answer tags.\n"
                    "If they aren't equivalent, there should only be 'NO' in the answer tags.\n"
                    "First answer: " + model_answer + "\n" + "Second answer: " + correct_answer
                )
            }
        ]
    }
    
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            resp_text = await resp.text()
            
            if resp.status == 200:
                data = await resp.json()
                model_response = data['choices'][0]['message']['content']
                # Extract <ANSWER>YES</ANSWER> or <ANSWER>NO</ANSWER>
                try:
                    responses_equivalent = model_response.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
                except:
                    responses_equivalent = "NO"

                final_answer = (responses_equivalent == "YES")
                
                async with aiofiles.open(check_file, 'a') as f:
                    await f.write("CORRECT ANSWER: " + correct_answer + '\n')
                    await f.write("MODEL OUTPUT: " + model_answer + '\n')
                    await f.write("COMPARISON ANSWER: ")
                    await f.write("TRUE\n" if final_answer else "FALSE\n")
                    await f.write(model_response + '\n')
                    await f.write("-" * 80 + "\n\n")

                return final_answer
            else:
                print(f"Error: {resp.status}")
                print(f"Error details: {resp_text}")
                return False
    except Exception as e:
        print(f"Exception in checking_answer: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def handle_question_async(
    session: aiohttp.ClientSession,
    question: str,
    answer: str,
    dataset: str,
    model: str = "openai/gpt-4",
    run_id: str = None,
    delay: int = 0
) -> bool:
    """
    Orchestrates getting the model's response for one question and checking correctness.
    Returns True if correct, False otherwise.
    """
    if delay > 0:
        print(f"    Waiting {delay}s before starting...")
        await asyncio.sleep(delay)
    
    print(f"    Getting model response...")    
    model_answer = await get_model_response_async(
        session, question, str(answer), dataset, model, run_id
    )
    
    if model_answer == "ERROR_RESPONSE":
        print(f"    Error getting model response")
        return False  # If we got an error, treat it as incorrect.
    
    print(f"    Checking answer: {model_answer}")
    result = await checking_answer_async(
        session, model_answer, str(answer), dataset, model=model, run_id=run_id
    )
    
    if result:
        print(f"    Answer CORRECT")
    else:
        print(f"    Answer INCORRECT")
        
    return result

async def run_accuracy_test_async(
    questions: list[str], 
    answers: list[str], 
    dataset: str, 
    model: str,
    max_samples: int = None,
    return_details: bool = True
) -> dict:
    """
    Asynchronously runs an accuracy test on a dataset with a specific model.
    
    Args:
        questions: List of questions to evaluate
        answers: List of correct answers
        dataset: Name of the dataset (gsm8k, math500, etc.)
        model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
        max_samples: Maximum number of samples to evaluate (useful for testing)
        return_details: Whether to return detailed results or just the accuracy score
        
    Returns:
        Dictionary containing the test results
    """
    # Create a unique run ID
    run_id = f"async_{dataset}"
    
    # Limit the number of samples if specified
    if max_samples is not None:
        questions = questions[:max_samples]
        answers = answers[:max_samples]
    
    print(f"RUNNING ASYNC ACCURACY TEST: {model} on {dataset}")
    print(f"Number of questions: {len(questions)}")
    print(f"Run ID: {run_id}")
    print("-" * 40)
    
    # Create directories
    base_dir = Path("results") / dataset
    output_dir = base_dir / "regular"
    metadata_dir = base_dir / "metadata"
    output_dir.mkdir(exist_ok=True, parents=True)
    metadata_dir.mkdir(exist_ok=True, parents=True)
    
    # We'll create the aiohttp session once and reuse it
    async with aiohttp.ClientSession() as session:
        # Create task list
        tasks = []
        for i in range(len(questions)):
            print(f"Setting up sample {i+1}/{len(questions)}")
            tasks.append(
                asyncio.create_task(
                    handle_question_async(
                        session,
                        questions[i],
                        answers[i],
                        dataset,
                        model=model,
                        run_id=run_id,
                        delay=i % 3  # Stagger requests slightly
                    )
                )
            )
        
        # Run all question checks concurrently
        print("=" * 40)
        print("Waiting for all tasks to complete...")
        results = await asyncio.gather(*tasks)
        
        # Count how many came back True
        total_correct = sum(results)
        accuracy = total_correct / len(questions)
        
        # Print results for each question
        print("\nRESULTS:")
        for i, is_correct in enumerate(results):
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"Sample {i+1}: {status}")
        
        # Build result summary
        wrong_indices = [i for i, correct in enumerate(results) if not correct]
        summary = {
            "accuracy": accuracy,
            "run_id": run_id,
            "model": model,
            "dataset": dataset,
            "num_questions": len(questions),
            "num_correct": total_correct,
            "wrong_indices": wrong_indices
        }
        
        # Add detailed results if requested
        if return_details:
            detailed_results = []
            for i, (question, answer, is_correct) in enumerate(zip(questions, answers, results)):
                detailed_results.append({
                    "index": i,
                    "question": question,
                    "correct_answer": str(answer),
                    "is_correct": is_correct
                })
            summary["results"] = detailed_results
        
        # Save the summary as JSON
        async with aiofiles.open(metadata_dir / f"accuracy_async_{model.split('/')[-1]}_{run_id}.json", 'w') as f:
            await f.write(json.dumps(summary, indent=2))
        
        print("-" * 40)
        print(f"ACCURACY for {model} on {dataset}: {accuracy:.4f}")
        print(f"Correct: {total_correct}/{len(questions)}")
        
        return summary

async def process_question_with_attempts(
    session: aiohttp.ClientSession,
    q_idx: int,
    question: str,
    answer: str,
    dataset: str,
    model: str,
    k: int,
    run_id: str,
    delay: int = 0
) -> dict:
    """
    Process a single question with k attempts, stopping if any attempt fails.
    This function processes attempts sequentially but multiple instances can run concurrently.
    
    Args:
        session: The aiohttp ClientSession to use
        q_idx: The index of the question (for logging)
        question: The question to evaluate
        answer: The correct answer
        dataset: Name of the dataset
        model: Model identifier
        k: Number of attempts to make
        run_id: Base run ID
        delay: Initial delay in seconds before starting this question's processing (default: 0)
        
    Returns:
        Dictionary with results for this question
    """
    question_result = {
        "passed": True,  # Assume passed until any attempt fails
        "attempts": []
    }
    
    # Apply initial delay if specified to stagger API calls
    if delay > 0:
        print(f"    Q{q_idx+1}: Waiting {delay}s before starting...")
        await asyncio.sleep(delay)
    
    print(f"\n>>> SAMPLE {q_idx+1} processing... <<<")
    
    # Make k attempts for this question
    for attempt in range(k):
        attempt_run_id = f"{run_id}_q{q_idx}_a{attempt}"
        
        print(f"  - Q{q_idx+1} Attempt {attempt+1}/{k} starting...")
        
        # Get the model's answer and check it
        is_correct = await handle_question_async(
            session,
            question,
            answer,
            dataset,
            model=model,
            run_id=attempt_run_id
        )
        
        question_result["attempts"].append(is_correct)
        
        # Print the result of this attempt
        if is_correct:
            print(f"  - Q{q_idx+1} Attempt {attempt+1}: ✓ CORRECT")
        else:
            print(f"  - Q{q_idx+1} Attempt {attempt+1}: ✗ WRONG")
        
        # In fail@k, if any attempt fails, the whole question fails
        if not is_correct:
            question_result["passed"] = False
            print(f"  ❌ Sample {q_idx+1} FAILED at attempt {attempt+1}")
            break
    
    if question_result["passed"]:
        print(f"  ✅ Sample {q_idx+1} PASSED (all {k} attempts correct)")
        
    return question_result

async def run_fail_at_k_test_async(
    questions: list[str], 
    answers: list[str], 
    dataset: str, 
    model: str,
    k: int = 4,
    max_samples: int = None,
    return_details: bool = True
) -> dict:
    """
    Asynchronously runs a fail@k test on a dataset with a specific model.
    In fail@k, a question is only considered correct if the model gets it right
    on ALL k attempts.
    
    Args:
        questions: List of questions to evaluate
        answers: List of correct answers
        dataset: Name of the dataset (gsm8k, math500, etc.)
        model: Model identifier (e.g., "anthropic/claude-3.5-sonnet")
        k: Number of attempts per question
        max_samples: Maximum number of samples to evaluate (useful for testing)
        return_details: Whether to return detailed results or just the accuracy score
        
    Returns:
        Dictionary containing the test results
    """
    # Create a unique run ID
    run_id = f"fail{k}_async_{dataset}"
    
    # Limit the number of samples if specified
    if max_samples is not None:
        questions = questions[:max_samples]
        answers = answers[:max_samples]
    
    print(f"RUNNING ASYNC FAIL@{k} TEST: {model} on {dataset}")
    print(f"Number of questions: {len(questions)}")
    print(f"Number of attempts per question: {k}")
    print(f"Run ID: {run_id}")
    print("-" * 40)
    
    # Create output directories
    base_dir = Path("results") / dataset
    output_dir = base_dir / "fail_at_k"
    metadata_dir = base_dir / "metadata"
    output_dir.mkdir(exist_ok=True, parents=True)
    metadata_dir.mkdir(exist_ok=True, parents=True)
    
    # We'll create the aiohttp session once and reuse it
    async with aiohttp.ClientSession() as session:
        # Create task list for processing questions concurrently
        tasks = []
        for q_idx in range(len(questions)):
            # Create a task for each question with staggered delays to avoid rate limits
            # Similar to the accuracy test, add a small delay based on index
            tasks.append(
                asyncio.create_task(
                    process_question_with_attempts(
                        session,
                        q_idx,
                        questions[q_idx],
                        answers[q_idx],
                        dataset,
                        model,
                        k,
                        run_id,
                        delay=q_idx % 3  # Stagger requests slightly like in the accuracy test
                    )
                )
            )
        
        print("=" * 40)
        print("Processing all questions concurrently...")
        # Run all question processing concurrently and collect results
        question_results = await asyncio.gather(*tasks)
        
        # Convert to the expected results format
        results = {}
        for q_idx, result in enumerate(question_results):
            results[f"q{q_idx}"] = result
    
    # Calculate fail@k score (questions that passed all attempts / total questions)
    passed_questions = sum(1 for r in results.values() if r["passed"])
    fail_at_k_score = passed_questions / len(questions)
    
    # Summarize results
    print("\nFAIL@K RESULTS SUMMARY:")
    for q_idx in range(len(questions)):
        question_id = f"q{q_idx}"
        status = "✅ PASSED" if results[question_id]["passed"] else "❌ FAILED"
        print(f"Sample {q_idx+1}: {status}")
    
    # Build result summary
    summary = {
        "fail_at_k_score": fail_at_k_score,
        "run_id": run_id,
        "model": model,
        "dataset": dataset,
        "k": k,
        "num_questions": len(questions),
        "num_passed": passed_questions,
        "results": results if return_details else None
    }
    
    # Save the summary as JSON
    async with aiofiles.open(metadata_dir / f"fail{k}_async_{model.split('/')[-1]}_{run_id}.json", 'w') as f:
        await f.write(json.dumps(summary, indent=2))
    
    print("-" * 40)
    print(f"FAIL@{k} score for {model} on {dataset}: {fail_at_k_score:.4f}")
    print(f"Passed: {passed_questions}/{len(questions)}")
    
    return summary

def visualize_results(dataset: str = None, run_id: str = None, save_figure: bool = True):
    """
    Visualizes results from accuracy and fail@k tests.
    
    Args:
        dataset: Name of the dataset to visualize (or None for all datasets)
        run_id: Specific run ID to visualize (or None for all runs)
        save_figure: Whether to save the figure to a file instead of displaying it
    """
    # Initialize data for visualization
    accuracy_data = []
    fail_at_k_data = []
    
    # Get all metadata directories
    base_dir = Path("results")
    if dataset:
        datasets = [dataset]
    else:
        datasets = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    for ds in datasets:
        metadata_dir = base_dir / ds / "metadata"
        if not metadata_dir.exists():
            continue
        
        # Process all JSON files in the metadata directory
        for json_file in metadata_dir.glob("*.json"):
            # Skip if a specific run_id was requested and this isn't it
            if run_id and run_id not in json_file.name:
                continue
                
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Add to appropriate dataset based on the file name
            if "accuracy" in json_file.name:
                accuracy_data.append({
                    "dataset": data["dataset"],
                    "model": data["model"],
                    "model_name": data["model"].split('/')[-1],
                    "accuracy": data["accuracy"],
                    "run_id": data["run_id"],
                    "metric": "Accuracy"
                })
            elif "fail" in json_file.name:
                fail_at_k_data.append({
                    "dataset": data["dataset"],
                    "model": data["model"],
                    "model_name": data["model"].split('/')[-1],
                    "score": data["fail_at_k_score"],
                    "k": data["k"],
                    "run_id": data["run_id"],
                    "metric": f"Fail@{data['k']}"
                })
    
    # Create DataFrames for visualization
    if accuracy_data:
        accuracy_df = pd.DataFrame(accuracy_data)
        print("== ACCURACY RESULTS ==")
        print(accuracy_df[["dataset", "model", "accuracy"]])
    
    if fail_at_k_data:
        fail_at_k_df = pd.DataFrame(fail_at_k_data)
        print("\n== FAIL@K RESULTS ==")
        print(fail_at_k_df[["dataset", "model", "k", "score"]])
    
    # Create combined visualization if both types of data exist
    if accuracy_data and fail_at_k_data:
        # Prepare data for combined chart
        combined_data = []
        
        # Add accuracy data
        for row in accuracy_data:
            combined_data.append({
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "metric": "Accuracy",
                "score": row["accuracy"]
            })
        
        # Add fail@k data
        for row in fail_at_k_data:
            combined_data.append({
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "metric": f"Fail@{row['k']}",
                "score": row["score"]
            })
        
        # Create DataFrame for combined chart
        combined_df = pd.DataFrame(combined_data)
        
        # For each dataset
        for ds in combined_df["dataset"].unique():
            ds_data = combined_df[combined_df["dataset"] == ds]
            
            # Set up the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Define bar width and group positions
            bar_width = 0.35
            models = sorted(ds_data["model_name"].unique())
            metrics = sorted(ds_data["metric"].unique())
            x = np.arange(len(models))
            
            # Plot bars for each model with the metrics side by side
            for i, model in enumerate(models):
                model_data = ds_data[ds_data["model_name"] == model]
                
                # Group by metric
                for j, metric in enumerate(metrics):
                    metric_model_data = model_data[model_data["metric"] == metric]
                    if not metric_model_data.empty:
                        score = metric_model_data["score"].values[0]
                        
                        # Calculate bar position - j is the position within the group
                        pos = i + (j - len(metrics)/2 + 0.5) * bar_width
                        
                        # Create bar with a different color for each metric
                        bar = ax.bar(pos, score, bar_width, 
                                     label=metric if i == 0 else "", 
                                     color=f'C{j}')
                        
                        # Add value label on top of the bar
                        height = score
                        ax.text(pos, height + 0.02, f'{height:.2f}', 
                                ha='center', va='bottom', fontsize=9)
            
            # Set plot labels and title
            ax.set_title(f"Performance Metrics for {ds}")
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Only include one legend entry per metric
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title="Metrics")
            
            ax.set_ylim(0, 1.15)  # Set y-axis to go from 0 to 1.15 to accommodate labels
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            if save_figure:
                # Create figures directory if it doesn't exist
                figures_dir = Path("results") / "figures"
                figures_dir.mkdir(exist_ok=True, parents=True)
                
                # Save figure
                figure_path = figures_dir / f"{ds}_combined_metrics.png"
                plt.savefig(figure_path)
                print(f"Figure saved to {figure_path}")
            else:
                plt.show()
    
    # Create individual plots if only one type of data exists
    elif accuracy_data:
        # Plot accuracy results
        for dataset in accuracy_df["dataset"].unique():
            dataset_df = accuracy_df[accuracy_df["dataset"] == dataset]
            models = sorted(dataset_df["model_name"].unique())
            x = np.arange(len(models))
            
            fig, ax = plt.subplots(figsize=(12, 6))
            scores = [dataset_df[dataset_df["model_name"] == model]["accuracy"].values[0] for model in models]
            
            bars = ax.bar(x, scores, 0.7)
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f"Accuracy for {dataset}")
            ax.set_xlabel("Model")
            ax.set_ylabel("Accuracy")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylim(0, 1.15)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            if save_figure:
                # Create figures directory if it doesn't exist
                figures_dir = Path("results") / "figures"
                figures_dir.mkdir(exist_ok=True, parents=True)
                
                # Save figure
                figure_path = figures_dir / f"{dataset}_accuracy.png"
                plt.savefig(figure_path)
                print(f"Figure saved to {figure_path}")
            else:
                plt.show()
    
    elif fail_at_k_data:
        # Plot fail@k results
        for dataset in fail_at_k_df["dataset"].unique():
            dataset_df = fail_at_k_df[fail_at_k_df["dataset"] == dataset]
            for k_val in dataset_df["k"].unique():
                k_df = dataset_df[dataset_df["k"] == k_val]
                models = sorted(k_df["model_name"].unique())
                x = np.arange(len(models))
                
                fig, ax = plt.subplots(figsize=(12, 6))
                scores = [k_df[k_df["model_name"] == model]["score"].values[0] for model in models]
                
                bars = ax.bar(x, scores, 0.7)
                
                # Add value labels on top of bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_title(f"Fail@{k_val} for {dataset}")
                ax.set_xlabel("Model")
                ax.set_ylabel(f"Fail@{k_val} Score")
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_ylim(0, 1.15)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                if save_figure:
                    # Create figures directory if it doesn't exist
                    figures_dir = Path("results") / "figures"
                    figures_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Save figure
                    figure_path = figures_dir / f"{dataset}_fail{k_val}.png"
                    plt.savefig(figure_path)
                    print(f"Figure saved to {figure_path}")
                else:
                    plt.show()
    
    if not accuracy_data and not fail_at_k_data:
        print("No results found for the specified criteria.")

async def main():
    """Main function to run experiments"""
    parser = argparse.ArgumentParser(description="Run fail@k evaluations")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math500", "gpqa"], 
                        help="Dataset to evaluate on")
    parser.add_argument("--models", type=str, nargs='+',
                        default=["anthropic/claude-3.5-sonnet", "openai/gpt-4o"],
                        help="Models to evaluate (space-separated list)")
    parser.add_argument("--max_samples", type=int, default=5, 
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--k", type=int, default=2, 
                        help="Number of attempts per question for fail@k")
    parser.add_argument("--test_type", type=str, default="both", choices=["accuracy", "fail_at_k", "both"],
                        help="Type of test to run")
    parser.add_argument("--clear_old", action="store_true",
                        help="Clear old results before running new tests")
    parser.add_argument("--clear_cache", action="store_true",
                        help="Clear cached datasets and re-download them")
    parser.add_argument("--single_model", type=str, default=None,
                        help="Run test for a single model instead of all models in the list")
    
    args = parser.parse_args()
    
    # Clear dataset cache if requested
    if args.clear_cache:
        cache_dir = Path("dataset_cache")
        if cache_dir.exists():
            print("Clearing dataset cache...")
            for cache_file in cache_dir.glob("*.pkl"):
                os.remove(cache_file)
                print(f"Removed {cache_file}")
    
    # Clear old results if requested
    if args.clear_old:
        metadata_dir = Path("results") / args.dataset / "metadata"
        if metadata_dir.exists():
            print(f"Clearing old results for {args.dataset}...")
            for json_file in metadata_dir.glob("*.json"):
                os.remove(json_file)
    
    # Select dataset
    if args.dataset == "gsm8k":
        questions = gsm8k_questions[:args.max_samples]
        answers = gsm8k_answers[:args.max_samples]
    elif args.dataset == "math500":
        questions = math500_questions[:args.max_samples]
        answers = math500_answers[:args.max_samples]
    elif args.dataset == "gpqa":
        questions = gpqa_questions[:args.max_samples]
        answers = gpqa_answers[:args.max_samples]
    
    # Determine which models to run
    models_to_run = [args.single_model] if args.single_model else args.models
    
    print(f"Running tests for {len(models_to_run)} models: {', '.join(models_to_run)}")
    print(f"Dataset: {args.dataset} with {len(questions)} questions")
    print(f"Test types: {args.test_type}")
    print("-" * 60)
    
    # Run experiments for each model
    for model in models_to_run:
        print(f"\n=== Testing model: {model} ===")
        
        # Run accuracy test if requested
        if args.test_type in ["accuracy", "both"]:
            await run_accuracy_test_async(
                questions,
                answers,
                args.dataset,
                model,
                max_samples=args.max_samples
            )
        
        # Run fail@k test if requested
        if args.test_type in ["fail_at_k", "both"]:
            await run_fail_at_k_test_async(
                questions,
                answers,
                args.dataset,
                model,
                k=args.k,
                max_samples=args.max_samples
            )
    
    # Visualize combined results
    visualize_results(dataset=args.dataset, save_figure=True)

if __name__ == "__main__":
    import argparse
    asyncio.run(main())
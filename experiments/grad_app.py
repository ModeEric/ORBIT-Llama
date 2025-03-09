import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import csv
import io

# Hugging Face API settings
API_KEY = "YOUR_HUGGINGFACE_API_KEY"  # Replace with your actual API key or use environment variables
MODELS = [
    {
        "name": "model-1-name",  # Replace with your actual model name
        "api_url": "https://api-url-for-model-1",  # Replace with the actual API URL for model 1
        "api_key": API_KEY
    },
    {
        "name": "model-2-name",  # Replace with your actual model name
        "api_url": "https://api-url-for-model-2",  # Replace with the actual API URL for model 2
        "api_key": API_KEY
    },
    {
        "name": "model-3-name",  # Replace with your actual model name
        "api_url": "https://api-url-for-model-3",  # Replace with the actual API URL for model 3
        "api_key": API_KEY
    }
]

def create_prompt(question):
    """Create a structured prompt with a conversational yet informative example answer."""
    return {
        "text": f"<|user|> **Question:** {question}\n\n<|assistant|> **Answer:**"
    }

def save_qa_pair(question, responses, ratings):
    """Save question, answers, and ratings to a JSON file with timestamp."""
    try:
        qa_record = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "responses": [
                {
                    "model_id": resp["model"],
                    "response": resp["response"],
                    "rating": rating["rating"]
                }
                for resp, rating in zip(responses, ratings)
            ]
        }
        
        try:
            with open("qa_history.json", "r") as f:
                qa_history = [json.loads(line) for line in f]
        except FileNotFoundError:
            qa_history = []
        
        with open("qa_history.json", "w") as f:
            for record in qa_history:
                json.dump(record, f)
                f.write("\n")
            json.dump(qa_record, f)
            f.write("\n")
                
        print(f"Saved Q&A record with timestamp {qa_record['timestamp']}")
        return True
    except Exception as e:
        print(f"Error saving Q&A record: {str(e)}")
        return False

def export_qa_history_to_csv():
    """Export Q&A history to CSV file, including ratings."""
    try:
        # Read the JSON history
        with open("qa_history.json", "r") as f:
            qa_history = [json.loads(line) for line in f]
        
        # Create a list to hold the flattened records
        flattened_records = []
        
        # Flatten the nested structure
        for record in qa_history:
            base_record = {
                "timestamp": record["timestamp"],
                "question": record["question"]
            }
            
            # Add responses and ratings from each model
            for response in record["responses"]:
                model_id = response["model_id"]
                response_text = response["response"]
                rating = response.get("rating", "")
                base_record[f"{model_id}_response"] = response_text
                base_record[f"{model_id}_rating"] = rating
            
            flattened_records.append(base_record)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_records)
        
        # Create a CSV string buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        return csv_buffer.getvalue()
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error exporting to CSV: {str(e)}")
        return None

def query_huggingface_api(api_url, api_key, question, model_name):
    try:
        prompt = create_prompt(question)
        print(f"\nQuerying {model_name}...")
        
        headers = {"Authorization": f"Bearer {api_key}"}
        
        if model_name == "specific-model-name":
            api_url = f"{api_url}/generate"
        
        payload = {
            "inputs": prompt["text"],
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "do_sample": True
            }
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()
        
        print(f"Raw response data from {model_name}:")
        print(response_data)
        
        if model_name == "specific-model-name":
            if isinstance(response_data, dict):
                response = response_data.get('generated_text', '').strip() + "<|endoftext|>"
            else:
                response = str(response_data).strip() + "<|endoftext|>"
        else:
            if isinstance(response_data, list) and len(response_data) > 0:
                if isinstance(response_data[0], dict):
                    response = response_data[0].get('generated_text', '').strip() + "<|endoftext|>"
                else:
                    response = str(response_data[0]).strip() + "<|endoftext|>"
            elif isinstance(response_data, dict):
                response = response_data.get('generated_text', '').strip() + "<|endoftext|>"
            else:
                response = str(response_data).strip() + "<|endoftext|>"
        
        print(f"Response received from {model_name}")
        return response
                
    except Exception as e:
        st.error(f"Error querying {model_name}: {str(e)}")
        print(f"Error details for {model_name}:", e)
        return f"Error: Failed to get response from {model_name}"

# Initialize session state for responses and ratings if they don't exist
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'ratings' not in st.session_state:
    st.session_state.ratings = {}

# Streamlit UI
st.title("Language Model Comparison Tool v2")

# Main question input and responses
user_input = st.text_area("Enter your astronomy question:")

if st.button("Get Responses"):
    if user_input:
        # Clear previous responses and ratings
        st.session_state.responses = []
        st.session_state.ratings = {}
        
        with st.spinner("Querying all models..."):
            responses = []

            for model in MODELS:
                response = query_huggingface_api(
                    model['api_url'], 
                    model['api_key'],
                    user_input,
                    model['name']
                )
                responses.append({"model": model["name"], "response": response})
                print(f"\nFull response from {model['name']}:")
                print(response[:200] + "...")
            
            st.session_state.responses = responses
            
            if save_qa_pair(user_input, responses, []):
                print("Successfully saved Q&A pairs")
            else:
                print("Failed to save Q&A pairs")

# Display responses if they exist
if st.session_state.responses:
    st.write("## Model Responses")
    ratings = []
    
    for idx, res in enumerate(st.session_state.responses):
        st.write(f"### Response {idx + 1}")
        st.write(res["response"])
        slider_key = f"rating_{idx}_{user_input}"
        rating = st.slider(f"Rate Response {idx + 1} (1-5)", 1, 5, 1, key=slider_key)
        ratings.append({"model": res["model"], "rating": rating})
    
    if st.button("Submit Ratings"):
        try:
            for idx, rating in enumerate(ratings):
                rating["question"] = user_input
                rating["response"] = st.session_state.responses[idx]["response"]
            
            # Save the Q&A pair along with ratings
            if save_qa_pair(user_input, st.session_state.responses, ratings):
                st.success("Ratings submitted successfully!")
                print("\nSubmitted ratings:")
                for rating in ratings:
                    print(f"Model: {rating['model']}, Rating: {rating['rating']}")
            else:
                st.error("Failed to save Q&A and ratings")
        except Exception as e:
            st.error(f"Error saving ratings: {str(e)}")

# History and Export section
st.markdown("---")
st.header("History and Export")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export Q&A History to CSV"):
        csv_data = export_qa_history_to_csv()
        if csv_data:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="qa_history.csv",
                mime="text/csv"
            )
            st.success("CSV file generated successfully!")
        else:
            st.error("No data available to export or an error occurred.")

with col2:
    show_history = st.button("Show Q&A History")

if show_history:
    st.header("Q&A History")
    try:
        with open("qa_history.json", "r") as f:
            qa_history = [json.loads(line) for line in f]
        
        for record in qa_history:
            st.write("---")
            st.write(f"**Question:** {record['question']}")
            st.write(f"**Time:** {record['timestamp']}")
            for response in record['responses']:
                st.write(f"\n**{response['model_id']}:**")
                st.write(response['response'])
                st.write(f"**Rating:** {response.get('rating', 'Not rated')}")
    except FileNotFoundError:
        st.write("No Q&A history available yet.")
    except Exception as e:
        st.error(f"Error loading Q&A history: {str(e)}")

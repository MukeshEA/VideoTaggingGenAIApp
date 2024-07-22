import streamlit as st
import pandas as pd
from openai import OpenAI
from langchain.prompts import PromptTemplate
import io

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["platform", "title", "videotitle", "description", "duration"],
    template="""Analyze the following {platform} video titled '{videotitle}' related to the game {title} and Description: {description} with a duration of {duration} in seconds
Based on the information passed, provide the one-word information from the given options for the attributes given below, separated by a pipe `|` after each information:
- Language
- Sentiment (positive, negative, neutral)
- Target audience (core gamer, casual gamer, non-gamer, or all)
- Target age (kid, teen, adult, or all)
- Target gender (male, female, or all)
- Creator type (game publisher, player)
- Main category and subcategory from ONLY below the given options: 
"Tutorial / Guide", "Walkthrough / Playthrough", "Highlight Reel / Montage", "High-level Gameplay", 
"Personalities / Vlogs", "Game Reviews / Impressions", "Game News / Updates", "Competitive Gaming / Esports", 
"Casual / Social Play", "Challenge / Achievement Hunting", "Speedrunning", "Lore / Story Analysis", "Mod Showcases", 
"Tech and Performance Analysis", "Game Development / Behind the Scenes", "Comparative Analysis", 
"Fan Content / Creations", "Streaming Highlights", "Interactive Play", "Educational / Historical Context", 
"Advertisements / Promotional Content", "Trailers"
Important Instructions:
If the video title indicates it is a trailer, the Creator type should be 'game publisher'
Choose the most appropriate Main category and Subcategory based on the video title and description
Respond ONLY with the following format, using EXACTLY these keys and separators, with NO additional text:
Language: <Language> | Sentiment: <Sentiment> | Target audience: <Target audience> | Target age: <Target age> | 
Target gender: <Target gender> | Creator type: <Creator type> | Main category: <Main category> | Subcategory: <Subcategory>"""
)

def extract_attributes(detailed_response):
    attributes = {}
    expected_keys = ['language', 'sentiment', 'target audience', 'target age', 'target gender', 'creator type', 'main category', 'subcategory']
    
    first_key = next((key for key in expected_keys if key in detailed_response.lower()), None)
    if first_key:
        detailed_response = detailed_response[detailed_response.lower().index(first_key):]
    
    key_value_pairs = detailed_response.split(' | ')
    for pair in key_value_pairs:
        parts = pair.split(':', 1)
        if len(parts) == 2:
            key, value = parts
            normalized_key = key.strip().lower()
            if normalized_key in expected_keys:
                attributes[normalized_key] = value.strip()
    
    for key in expected_keys:
        if key not in attributes:
            attributes[key] = ""
    
    return attributes

def normalize_language(language):
    if language.lower() in ["en", "english"]:
        return "en"
    return language

def analyze_video(client, prompt_template, platform, title, videotitle, description, duration, max_retries=3):
    prompt = prompt_template.format(
        platform=platform, title=title, videotitle=videotitle, description=description, duration=duration
    )
    print(f"Analyzing video: {videotitle}")
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="",
                max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.3
            )
            detailed_response = response.choices[0].message.content.strip()
            attributes = extract_attributes(detailed_response)
            
            if all(attributes.get(key, "") for key in ['language', 'sentiment', 'target audience', 'target age', 'target gender', 'creator type', 'main category', 'subcategory']):
                print(f"Response for '{videotitle}': {detailed_response}")
                return detailed_response
            
            print(f"Attempt {attempt + 1} failed. Invalid format. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Error: {str(e)}. Retrying...")
    
    print(f"Failed to get valid response after {max_retries} attempts for video: {videotitle}")
    return ""

def tag_videos(client, df, prompt_template):
    required_columns = ['Title', 'Platform', 'VideoTitle', 'Description', 'Duration']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The input DataFrame must contain columns: {', '.join(required_columns)}.")
    
    new_columns = ['Detailed Response', 'Language', 'Sentiment', 'Target Audience', 
                   'Target Age', 'Target Gender', 'Creator Type', 'Main Category', 'Sub Category']
    
    for col in new_columns:
        df[col] = ""

    for index, row in df.iterrows():
        try:
            detailed_response = analyze_video(client, prompt_template, row['Platform'], row['Title'], row['VideoTitle'], row['Description'], row['Duration'])
            attributes = extract_attributes(detailed_response)
            
            df.at[index, 'Detailed Response'] = detailed_response
            df.at[index, 'Language'] = normalize_language(attributes.get("language", ""))
            df.at[index, 'Sentiment'] = attributes.get("sentiment", "")
            df.at[index, 'Target Audience'] = attributes.get("target audience", "")
            df.at[index, 'Target Age'] = attributes.get("target age", "")
            df.at[index, 'Target Gender'] = attributes.get("target gender", "")
            df.at[index, 'Creator Type'] = attributes.get("creator type", "")
            df.at[index, 'Main Category'] = attributes.get("main category", "")
            df.at[index, 'Sub Category'] = attributes.get("subcategory", "")
            
        except Exception as e:
            print(f"Error processing row {index}: {row['VideoTitle']}. Error: {str(e)}")
    
    return df

def excel_analysis(client, prompt_template):
    st.header("Excel File Analysis")
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("Uploaded DataFrame:")
        st.dataframe(df.head())

        if st.button("Analyze Videos"):
            with st.spinner("Analyzing videos..."):
                tagged_df = tag_videos(client, df, prompt_template)
            
            st.success("Analysis complete!")
            st.write("Results:")
            st.dataframe(tagged_df)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                tagged_df.to_excel(writer, index=False)
            output.seek(0)
            st.download_button(
                label="Download Results",
                data=output,
                file_name="analyzed_videos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def manual_input_analysis(client, prompt_template):
    st.header("Manual Input Analysis")
    
    platform = st.text_input("Platform")
    title = st.text_input("Title")
    videotitle = st.text_input("Video Title")
    description = st.text_area("Description")
    duration = st.number_input("Duration", min_value=0.0, step=0.1)

    if st.button("Analyze Video"):
        if all([platform, title, videotitle, description, duration]):
            with st.spinner("Analyzing video..."):
                detailed_response = analyze_video(client, prompt_template, platform, title, videotitle, description, duration)
            
            if detailed_response:
                st.success("Analysis complete!")
                st.write("Results:")
                attributes = extract_attributes(detailed_response)
                for key, value in attributes.items():
                    st.write(f"{key.capitalize()}: {value}")
            else:
                st.error("Failed to analyze the video. Please try again.")
        else:
            st.warning("Please fill in all fields before analyzing.")

def main():
    st.title("Video Analysis App")

    HOST = "http://dev-eadp-ai-llm.data.ea.com/llama3"
    client = OpenAI(base_url=f"{HOST}/v1", api_key="-")

    tab1, tab2 = st.tabs(["Upload Excel", "Manual Input"])

    with tab1:
        excel_analysis(client, prompt_template)

    with tab2:
        manual_input_analysis(client, prompt_template)

if __name__ == "__main__":
    main()
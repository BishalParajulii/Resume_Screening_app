import nltk
import pickle as pk
import streamlit as st
import re

nltk.download('punkt')
nltk.download('stopwords')

#Loading models
clk = pk.load(open('clf.pkl' , 'rb'))
tfidf = pk.load(open('tfidf.pkl' , 'rb'))


def cleanResume(txt):
    # Remove URLs
    cleanTxt = re.sub(r'http\S+', '', txt)

    # Remove emails
    cleanTxt = re.sub(r'\S+@\S+', '', cleanTxt)

    # Remove hashtags
    cleanTxt = re.sub(r'#\S+', '', cleanTxt)

    # Remove special characters
    cleanTxt = re.sub(r'[^A-Za-z0-9\s]', '', cleanTxt)

    # Remove carriage return and newline sequences
    cleanTxt = re.sub(r'\r\n', '', cleanTxt)

    return cleanTxt


#web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload Your Resume" , type = ['txt' , 'pdf'])

    if uploaded_file is not None:
        try:
            resume_byte = uploaded_file.read()
            resume_text = resume_byte.decode('utf-8')
        except UnicodeError:
            resume_text = resume_byte.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        pred_id = clk.predict(cleaned_resume)[0]
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(pred_id  , "Unknown")
        st.write(category_name)



#python main
if __name__ == "__main__":
    main()




import streamlit as st

def main():
    st.title("Risk Analysis System")
    text = st.text_area("Enter Claim Description:")
    if st.button("Analyze"):
        result = analyze_risk(text)
        st.write(result)

if __name__ == "__main__":
    main()

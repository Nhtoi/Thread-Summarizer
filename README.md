<p align="center"> 
  <img src="https://img.shields.io/badge/Status-Finished-green" alt="Project Status">
  <img src="https://img.shields.io/badge/Python-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-red?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-black?logo=flask&logoColor=white" alt="Flask">
</p>
Reddit Thread Summarizer
This application uses Flask to provide a web interface for summarizing Reddit threads. It leverages a fine-tuned text summarization model (T5 by default) to generate concise summaries of thread comments.
🚀 How It Works
The application connects app.py (web interface) with test.py (summarization engine).

When a user submits a Reddit thread URL:
Comments are extracted
Text is cleaned and processed
A summary is generated using a fine-tuned model
The summary is displayed on the results page

Much More Detailed Explanation: https://docs.google.com/document/d/1Jv6_a2ziVCMMDE7UnZNm-4ushzFUhwiK/edit?usp=sharing&ouid=101291386082869297856&rtpof=true&sd=true


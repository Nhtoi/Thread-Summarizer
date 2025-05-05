from flask import Flask, request, render_template, redirect, url_for
import os
import re
import praw
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    username=os.getenv("USERNAME"),
    password=os.getenv("PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
)

def extract_comments_from_thread(url, score_threshold=5, limit=100):
    """Extract comments from a Reddit thread."""
    submission = reddit.submission(url=url)
    print(f"\n Extracting from: {submission.title}")

    submission.comments.replace_more(limit=None)
    comments_list = submission.comments.list()

    comments_data = []

    for comment in comments_list[:limit]:
        comment_body = comment.body.replace("\n", " ").strip()
        comment_score = comment.score
        if comment_score < score_threshold:
            continue

        comments_data.append({
            "comment_body": comment_body,
            "comment_score": comment_score
        })

    return comments_data

def clean_text(text):
    """Clean and prepare text."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove Reddit formatting
    text = re.sub(r'\[.*?\]', '', text)
    # Remove TL;DR markers
    text = re.sub(r'TL;DR|TLDR|tl;dr|tldr', '', text, flags=re.IGNORECASE)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', '', text)
    # Simplify whitespace
    text = ' '.join(text.split())

    return text.strip()

def simple_summarize(comments, max_length=200):
    """A simple summarization function that doesn't rely on ML models.
    Just extracts the highest scored comments up to a certain length."""
    
    # Sort comments by score (highest first)
    sorted_comments = sorted(comments, key=lambda x: x["comment_score"], reverse=True)
    
    # Take top comments until we reach max_length
    summary = ""
    for comment in sorted_comments:
        clean_comment = clean_text(comment["comment_body"])
        if len(summary) + len(clean_comment) + 3 <= max_length:
            if summary:
                summary += " - " + clean_comment
            else:
                summary = clean_comment
        else:
            if not summary:
                # If we don't have any summary yet, take at least part of the top comment
                summary = clean_comment[:max_length-3] + "..."
            break
    
    return summary if summary else "No meaningful summary could be generated."

@app.route('/', methods=['GET', 'POST'])
def get_link():
    if request.method == 'POST':
        link = request.form['Link']
        # Save the link for reference
        with open('link.txt', 'w') as f:
            f.write(link)
        
        try:
            # Process the link and generate summary
            comments = extract_comments_from_thread(link)
            
            if not comments:
                return render_template('error.html', message="No comments above the score threshold.")
            
            # Use simple summarization
            summary = simple_summarize(comments)
            return redirect(url_for('summary_result', summary=summary))
        
        except Exception as e:
            return render_template('error.html', message=f"Error: {str(e)}")
    
    return render_template('home.html')

@app.route('/summary', methods=['GET'])
def summary_result():
    summary = request.args.get('summary', '')
    return render_template('summary.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
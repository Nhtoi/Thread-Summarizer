from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Initialize these to None for now
model = None
tokenizer = None
model_type = "facebook/bart-base"  # You can change this to your preferred model
base_model_type = "bart"  # Change to match your model type

def init_model():
    """Initialize the model lazily when needed."""
    global model, tokenizer, model_type, base_model_type
    
    if model is not None and tokenizer is not None:
        return True
        
    try:
        # Try to import required modules - if they fail, we'll catch the exception
        from test import generate_summary, get_model_and_tokenizer, clean_text, load_fine_tuned_model
        
        # Try to load fine-tuned model first
        try:
            model, tokenizer = load_fine_tuned_model(model_type)
            print("Successfully loaded fine-tuned model")
            return True
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            
        # If that fails, try to load pretrained model
        try:
            model, tokenizer = get_model_and_tokenizer(model_type)
            print("Successfully loaded pretrained model")
            return True
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            return False
            
    except ImportError as e:
        print(f"Import error: {e}")
        return False

@app.route('/', methods=['GET', 'POST'])
def get_link():
    if request.method == 'POST':
        link = request.form['Link']
        # Save the link for reference
        with open('link.txt', 'w') as f:
            f.write(link)
        
        # First check if we can initialize the model
        if not init_model():
            return render_template('error.html', 
                message="Model initialization failed. Please make sure all required libraries are installed: pip install sentencepiece transformers")
        
        # Import here to avoid issues if modules are missing
        try:
            from test import extract_comments_from_thread, generate_summary
            
            # Process the link and generate summary
            comments = extract_comments_from_thread(link)
            
            if not comments:
                return render_template('error.html', message="No comments above the score threshold.")
            
            full_text = " ".join([c["comment_body"] for c in comments])
            
            if len(full_text.strip()) < 10:
                return render_template('error.html', message="Not enough content to summarize.")
            
            summary = generate_summary(full_text, model, tokenizer, model_type=base_model_type)
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
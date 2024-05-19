from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model and data
def load_model():
    with open('model01.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
df = data['df']
model = data['model']
cosine_sim = data['cosine_sim']



# Content-based recommendation function
def get_content_based_recommendations(product_id, cosine_sim=cosine_sim):
    idx = df.index[df['Product ID'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    product_indices = [i[0] for i in sim_scores]
    return df['Product ID'].iloc[product_indices[1:11]].tolist()



# Hybrid recommendation function
def hybrid_recommendations(user_id, top_n=5, model=model):
    user_ratings = df[df['User ID'] == user_id]['Product ID']
    all_product_ids = df['Product ID'].unique()
    predictions = [model.predict(user_id, pid) for pid in all_product_ids if pid not in user_ratings.values]
    cf_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
    top_product_id = cf_recommendations[0].iid
    content_recommendations = get_content_based_recommendations(top_product_id)
    combined_recommendations = [pred.iid for pred in cf_recommendations] + content_recommendations
    combined_recommendations = list(dict.fromkeys(combined_recommendations))[:top_n]
    return combined_recommendations




# Define API routes
@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    user_id = data['user_id']
    hybrid_recs = hybrid_recommendations(user_id)
    return jsonify({"recommendations": hybrid_recs})

if __name__ == '__main__':
    app.run(debug=True)

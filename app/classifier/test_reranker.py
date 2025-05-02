from classifier_reranker import ClassifierReranker

reranker = ClassifierReranker()
score = reranker.predict_match_score("React developer with 5 years experience", "We need a React engineer for UI migration.")
print("🔚 Final Score:", score)
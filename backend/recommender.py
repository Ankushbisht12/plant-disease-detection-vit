def get_recommendation(crop: str, disease: str) -> str:
    crop = crop.lower()
    disease = disease.lower()

    if crop == "apple":
        if "scab" in disease:
            return "Apply fungicide like Mancozeb. Remove infected leaves and improve air circulation."
        if "black_rot" in disease:
            return "Prune infected branches. Apply copper-based fungicide."
        return "Maintain orchard hygiene and monitor regularly."

    if crop == "potato":
        if "early_blight" in disease:
            return "Apply chlorothalonil fungicide. Avoid overhead irrigation."
        if "late_blight" in disease:
            return "Urgent: apply metalaxyl-based fungicide and destroy infected plants."
        return "Use certified seed potatoes and rotate crops."

    if crop == "tomato":
        if "leaf_curl" in disease or "yellow" in disease:
            return "Control whiteflies using neem oil or imidacloprid."
        if "bacterial" in disease:
            return "Remove infected plants and avoid working in wet fields."
        if "early_blight" in disease or "late_blight" in disease:
            return "Apply appropriate fungicide and ensure good drainage."
        return "Ensure balanced fertilization and proper spacing."

    return "No specific recommendation available. Consult agricultural expert."

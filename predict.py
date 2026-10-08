# ============================================================
# EXPLAINABLE AI BASED INTRUSION DETECTION
# PREDICTION USING RANDOM FOREST + SHAP
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# ============================================================
# 1. MODEL PATHS
# ============================================================

BINARY_MODEL_PATH = "models/random_forest_binary.pkl"
ATTACK_MODEL_PATH = "models/random_forest_attack_type.pkl"
FEATURE_PATH = "models/selected_features.pkl"


# ============================================================
# 2. CHECK MODEL FILES
# ============================================================

required_files = [
    BINARY_MODEL_PATH,
    ATTACK_MODEL_PATH,
    FEATURE_PATH
]

for file_path in required_files:
    if not os.path.exists(file_path):
        print("\nERROR:")
        print(f"Required file not found: {file_path}")
        print("Please run train.py first.")
        raise SystemExit


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

print("\n==============================================")
print("LOADING TRAINED MODELS")
print("==============================================")

binary_model = joblib.load(BINARY_MODEL_PATH)

attack_model = joblib.load(ATTACK_MODEL_PATH)

FEATURES = joblib.load(FEATURE_PATH)

print("Binary Random Forest loaded successfully.")
print("Attack Type Random Forest loaded successfully.")
print("Selected features loaded successfully.")

print("\nNumber of Input Features:", len(FEATURES))


# ============================================================
# 4. DISPLAY FEATURES
# ============================================================

print("\n==============================================")
print("REQUIRED INPUT FEATURES")
print("==============================================")

for i, feature in enumerate(FEATURES, start=1):
    print(f"{i:02d}. {feature}")


# ============================================================
# 5. CONFIDENCE LEVEL FUNCTION
# ============================================================

def get_confidence_level(confidence):

    if confidence >= 80:
        return "HIGH"

    elif confidence >= 60:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# 6. SAFE FLOAT INPUT FUNCTION
# ============================================================

def get_numeric_input(feature_name):

    while True:

        try:

            value = input(
                f"Enter {feature_name}: "
            ).strip()

            # Empty input = 0
            if value == "":
                return 0.0

            return float(value)

        except ValueError:

            print(
                "Invalid input. "
                "Please enter a numeric value."
            )


# ============================================================
# 7. COLLECT 20 INPUT VALUES
# ============================================================

def collect_input():

    print("\n==============================================")
    print("ENTER NETWORK TRAFFIC VALUES")
    print("==============================================")

    print(
        "\nEnter the 20 network feature values."
    )

    print(
        "Press Enter without typing a value "
        "to use 0.\n"
    )

    input_data = {}

    for feature in FEATURES:

        input_data[feature] = get_numeric_input(
            feature
        )

    # Convert to DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    return input_df


# ============================================================
# 8. SHAP EXPLANATION FUNCTION
# ============================================================

def get_shap_explanation(
    model,
    input_df,
    predicted_class,
    top_n=5
):

    if not SHAP_AVAILABLE:

        return {
            "available": False,
            "message":
                "SHAP package is not installed."
        }

    try:

        # TreeExplainer is suitable for Random Forest
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            input_df
        )

        feature_values = None


        # ----------------------------------------------------
        # CASE 1:
        # Older SHAP versions:
        # shap_values = list
        # one array per class
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            classes = list(model.classes_)

            if predicted_class in classes:

                class_index = classes.index(
                    predicted_class
                )

            else:

                class_index = 0

            feature_values = np.array(
                shap_values[class_index]
            )[0]


        # ----------------------------------------------------
        # CASE 2:
        # Newer SHAP versions may return ndarray
        # ----------------------------------------------------

        else:

            shap_array = np.array(
                shap_values
            )


            # Example:
            # (samples, features, classes)
            if shap_array.ndim == 3:

                classes = list(model.classes_)

                if predicted_class in classes:

                    class_index = classes.index(
                        predicted_class
                    )

                else:

                    class_index = 0

                feature_values = shap_array[
                    0,
                    :,
                    class_index
                ]


            # Example:
            # (samples, features)
            elif shap_array.ndim == 2:

                feature_values = shap_array[0]


            else:

                feature_values = (
                    shap_array
                    .reshape(-1)
                )


        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        feature_values = np.array(
            feature_values
        ).reshape(-1)

        number_of_features = min(
            len(FEATURES),
            len(feature_values)
        )


        explanation = []

        for i in range(number_of_features):

            shap_value = float(
                feature_values[i]
            )

            explanation.append({

                "feature":
                    FEATURES[i],

                "input_value":
                    float(
                        input_df.iloc[0][
                            FEATURES[i]
                        ]
                    ),

                "shap_value":
                    shap_value,

                "impact":
                    abs(shap_value)

            })


        # Sort highest contribution first
        explanation = sorted(

            explanation,

            key=lambda x: x["impact"],

            reverse=True
        )


        return {

            "available": True,

            "top_features":
                explanation[:top_n]

        }


    except Exception as error:

        return {

            "available": False,

            "message":
                str(error)

        }


# ============================================================
# 9. PREDICTION FUNCTION
# ============================================================

def predict_network(input_df):

    # --------------------------------------------------------
    # BINARY PREDICTION
    # --------------------------------------------------------

    binary_prediction = int(
        binary_model.predict(
            input_df
        )[0]
    )


    binary_probabilities = (
        binary_model.predict_proba(
            input_df
        )[0]
    )


    binary_classes = list(
        binary_model.classes_
    )


    predicted_binary_index = (
        binary_classes.index(
            binary_prediction
        )
    )


    binary_confidence = float(
        binary_probabilities[
            predicted_binary_index
        ] * 100
    )


    confidence_level = (
        get_confidence_level(
            binary_confidence
        )
    )


    # ========================================================
    # NORMAL TRAFFIC
    # ========================================================

    if binary_prediction == 0:

        detection_status = "NORMAL"

        attack_type = "Normal"

        attack_confidence = None


        shap_result = (
            get_shap_explanation(

                binary_model,

                input_df,

                binary_prediction
            )
        )


    # ========================================================
    # ATTACK TRAFFIC
    # ========================================================

    else:

        detection_status = "ATTACK"


        # ----------------------------------------------------
        # ATTACK TYPE PREDICTION
        # ----------------------------------------------------

        attack_prediction = (
            attack_model.predict(
                input_df
            )[0]
        )


        attack_probabilities = (
            attack_model.predict_proba(
                input_df
            )[0]
        )


        attack_classes = list(
            attack_model.classes_
        )


        attack_index = (
            attack_classes.index(
                attack_prediction
            )
        )


        attack_confidence = float(

            attack_probabilities[
                attack_index
            ] * 100

        )


        attack_type = str(
            attack_prediction
        )


        # ----------------------------------------------------
        # SHAP FOR ATTACK TYPE MODEL
        # ----------------------------------------------------

        shap_result = (
            get_shap_explanation(

                attack_model,

                input_df,

                attack_prediction
            )
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    result = {

        "detection_status":
            detection_status,

        "attack_type":
            attack_type,

        "confidence":
            round(
                binary_confidence,
                2
            ),

        "confidence_level":
            confidence_level,

        "attack_confidence":
            (
                round(
                    attack_confidence,
                    2
                )
                if attack_confidence
                is not None
                else None
            ),

        "shap":
            shap_result
    }


    return result


# ============================================================
# 10. DISPLAY RESULT
# ============================================================

def display_result(result):

    print("\n\n")
    print("=" * 60)
    print("INTRUSION DETECTION RESULT")
    print("=" * 60)


    print(
        f"\nDetection Status : "
        f"{result['detection_status']}"
    )


    print(
        f"Attack Type      : "
        f"{result['attack_type']}"
    )


    print(
        f"Confidence       : "
        f"{result['confidence']:.2f}%"
    )


    print(
        f"Confidence Level : "
        f"{result['confidence_level']}"
    )


    # Attack-type confidence
    if (
        result["attack_confidence"]
        is not None
    ):

        print(
            f"Attack Confidence: "
            f"{result['attack_confidence']:.2f}%"
        )


    # ========================================================
    # SHAP OUTPUT
    # ========================================================

    print("\n")
    print("-" * 60)
    print("SHAP EXPLANATION")
    print("-" * 60)


    shap_result = result["shap"]


    if shap_result["available"]:

        print(
            "\nTop Features Influencing "
            "the Prediction:\n"
        )


        for rank, item in enumerate(
            shap_result["top_features"],
            start=1
        ):

            shap_value = (
                item["shap_value"]
            )


            # Positive or negative influence
            if shap_value > 0:

                direction = (
                    "Positive Influence"
                )

            elif shap_value < 0:

                direction = (
                    "Negative Influence"
                )

            else:

                direction = (
                    "Neutral"
                )


            print(
                f"{rank}. "
                f"{item['feature']}"
            )

            print(
                f"   Input Value : "
                f"{item['input_value']}"
            )

            print(
                f"   SHAP Value  : "
                f"{shap_value:.6f}"
            )

            print(
                f"   Impact      : "
                f"{direction}"
            )

            print()


    else:

        print(
            "\nSHAP explanation "
            "could not be generated."
        )

        print(
            "Reason:",
            shap_result["message"]
        )


    print("=" * 60)


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("AI BASED NETWORK INTRUSION DETECTION")
    print("RANDOM FOREST + SHAP")
    print("=" * 60)


    # Collect 20 inputs
    input_df = collect_input()


    print("\n==============================================")
    print("INPUT DATA")
    print("==============================================")

    print(
        input_df.to_string(
            index=False
        )
    )


    print("\nAnalyzing network traffic...")


    # Prediction
    result = predict_network(
        input_df
    )


    # Display
    display_result(
        result
    )


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":

    main()
# ============================================================
# EXPLAINABLE AI BASED INTRUSION DETECTION
# FLASK WEB APPLICATION
# Random Forest + SHAP + SQLite
# ============================================================

import os
import sqlite3
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)


# ============================================================
# 1. SHAP IMPORT
# ============================================================

try:
    import shap
    SHAP_AVAILABLE = True

    print("SHAP Version:", shap.__version__)

except ImportError:

    SHAP_AVAILABLE = False

    print(
        "WARNING: SHAP is not installed. "
        "Run: pip install shap"
    )


# ============================================================
# 2. FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "intrusion_detection_secret_key_2026"


# ============================================================
# 3. BASE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "intrusion_detection.db"
)


# ============================================================
# 4. MODEL PATHS
# ============================================================

BINARY_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_binary.pkl"
)

ATTACK_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "random_forest_attack_type.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "selected_features.pkl"
)


# ============================================================
# 5. CHECK MODEL FILES
# ============================================================

required_files = [
    BINARY_MODEL_PATH,
    ATTACK_MODEL_PATH,
    FEATURE_PATH
]


for file_path in required_files:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Required model file not found: {file_path}"
        )


# ============================================================
# 6. LOAD MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODELS")
print("=" * 70)


binary_model = joblib.load(
    BINARY_MODEL_PATH
)

attack_model = joblib.load(
    ATTACK_MODEL_PATH
)

FEATURES = joblib.load(
    FEATURE_PATH
)


print("Binary Random Forest     : Loaded")
print("Attack Type Random Forest: Loaded")
print("Selected Features        : Loaded")
print("Number of Features       :", len(FEATURES))

print("\nSelected Features:")

for index, feature in enumerate(
    FEATURES,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )

print("=" * 70)


# ============================================================
# 7. DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# 8. INITIALIZE DATABASE
# ============================================================

def initialize_database():

    connection = get_db_connection()

    cursor = connection.cursor()


    # --------------------------------------------------------
    # ADMIN TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS admin (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            username TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL
        )
        """
    )


    # --------------------------------------------------------
    # PREDICTIONS TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            input_values TEXT,

            detection_status TEXT,

            attack_type TEXT,

            confidence REAL,

            confidence_level TEXT,

            attack_confidence REAL,

            shap_features TEXT,

            prediction_date TEXT
        )
        """
    )


    # --------------------------------------------------------
    # DEFAULT ADMIN
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT *
        FROM admin
        WHERE username = ?
        """,
        ("admin",)
    )


    admin = cursor.fetchone()


    if admin is None:

        cursor.execute(
            """
            INSERT INTO admin (
                username,
                password
            )
            VALUES (?, ?)
            """,
            (
                "admin",
                "admin123"
            )
        )


    connection.commit()

    connection.close()


initialize_database()


# ============================================================
# 9. CONFIDENCE LEVEL
# ============================================================

def get_confidence_level(
    confidence
):

    if confidence >= 80:

        return "HIGH"

    elif confidence >= 60:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# 10. SHAP EXPLANATION
# CORRECTED VERSION FOR NEW + OLD SHAP FORMATS
# ============================================================

def get_shap_explanation(
    model,
    input_df,
    predicted_class,
    top_n=5
):

    # --------------------------------------------------------
    # CHECK SHAP
    # --------------------------------------------------------

    if not SHAP_AVAILABLE:

        print(
            "\nSHAP ERROR: "
            "SHAP package is not installed."
        )

        return []


    try:

        print("\n")
        print("=" * 60)
        print("SHAP DEBUG")
        print("=" * 60)

        print(
            "Predicted Class :",
            predicted_class
        )

        print(
            "Model Classes   :",
            model.classes_
        )

        print(
            "Input Shape     :",
            input_df.shape
        )


        # ----------------------------------------------------
        # CREATE TREE EXPLAINER
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            model
        )


        # ----------------------------------------------------
        # PREFERRED NEW SHAP API
        # ----------------------------------------------------

        try:

            explanation = explainer(
                input_df
            )

            values = np.asarray(
                explanation.values
            )

            print(
                "SHAP API       : New API"
            )


        # ----------------------------------------------------
        # FALLBACK OLD SHAP API
        # ----------------------------------------------------

        except Exception as new_api_error:

            print(
                "New SHAP API failed:",
                new_api_error
            )

            print(
                "Trying old SHAP API..."
            )


            old_values = (
                explainer.shap_values(
                    input_df
                )
            )


            # ------------------------------------------------
            # OLD API RETURNS LIST
            # ------------------------------------------------

            if isinstance(
                old_values,
                list
            ):

                classes = list(
                    model.classes_
                )


                class_index = next(

                    (
                        index

                        for index, cls
                        in enumerate(classes)

                        if str(cls)
                        ==
                        str(predicted_class)
                    ),

                    0
                )


                values = np.asarray(
                    old_values[
                        class_index
                    ]
                )


            else:

                values = np.asarray(
                    old_values
                )


            print(
                "SHAP API       : Old API"
            )


        # ----------------------------------------------------
        # DISPLAY SHAPE
        # ----------------------------------------------------

        print(
            "SHAP Shape     :",
            values.shape
        )


        # ----------------------------------------------------
        # MODEL CLASSES
        # ----------------------------------------------------

        classes = list(
            model.classes_
        )


        # ----------------------------------------------------
        # FIND PREDICTED CLASS INDEX
        # ----------------------------------------------------

        class_index = next(

            (
                index

                for index, cls
                in enumerate(classes)

                if str(cls)
                ==
                str(predicted_class)
            ),

            0
        )


        print(
            "Class Index    :",
            class_index
        )


        # ====================================================
        # EXTRACT FEATURE SHAP VALUES
        # ====================================================

        feature_values = None


        # ----------------------------------------------------
        # CASE 1
        #
        # Shape:
        # (samples, features, classes)
        #
        # Example:
        # (1, 20, 14)
        # ----------------------------------------------------

        if values.ndim == 3:

            if (
                values.shape[1]
                ==
                len(FEATURES)

                and

                values.shape[2]
                ==
                len(classes)
            ):

                feature_values = values[
                    0,
                    :,
                    class_index
                ]


            # ------------------------------------------------
            # Shape:
            # (samples, classes, features)
            # ------------------------------------------------

            elif (
                values.shape[1]
                ==
                len(classes)

                and

                values.shape[2]
                ==
                len(FEATURES)
            ):

                feature_values = values[
                    0,
                    class_index,
                    :
                ]


            else:

                raise ValueError(
                    "Unexpected 3D SHAP shape: "
                    f"{values.shape}"
                )


        # ----------------------------------------------------
        # CASE 2
        #
        # Shape:
        # (samples, features)
        # ----------------------------------------------------

        elif values.ndim == 2:

            # Standard:
            # 1 x 20

            if (
                values.shape[0] == 1
                and
                values.shape[1]
                == len(FEATURES)
            ):

                feature_values = (
                    values[0]
                )


            # Alternative:
            # 20 x 1

            elif (
                values.shape[0]
                == len(FEATURES)

                and

                values.shape[1]
                == 1
            ):

                feature_values = (
                    values[:, 0]
                )


            else:

                # Last-resort flatten
                flattened = (
                    values.reshape(-1)
                )

                if (
                    len(flattened)
                    ==
                    len(FEATURES)
                ):

                    feature_values = (
                        flattened
                    )

                else:

                    raise ValueError(
                        "Unexpected 2D SHAP shape: "
                        f"{values.shape}"
                    )


        # ----------------------------------------------------
        # CASE 3
        #
        # Shape:
        # (features,)
        # ----------------------------------------------------

        elif values.ndim == 1:

            feature_values = values


        else:

            raise ValueError(
                "Unsupported SHAP shape: "
                f"{values.shape}"
            )


        # ====================================================
        # FINAL CONVERSION
        # ====================================================

        feature_values = np.asarray(
            feature_values,
            dtype=float
        ).reshape(-1)


        print(
            "Extracted Shape :",
            feature_values.shape
        )


        # ----------------------------------------------------
        # VERIFY 20 FEATURES
        # ----------------------------------------------------

        if (
            len(feature_values)
            !=
            len(FEATURES)
        ):

            raise ValueError(

                "SHAP feature count mismatch. "

                f"Expected {len(FEATURES)}, "

                f"received "
                f"{len(feature_values)}."
            )


        # ====================================================
        # CREATE SHAP EXPLANATION
        # ====================================================

        explanations = []


        for index, feature_name in enumerate(
            FEATURES
        ):

            shap_value = float(
                feature_values[
                    index
                ]
            )


            input_value = float(
                input_df.iloc[
                    0
                ][
                    feature_name
                ]
            )


            # ------------------------------------------------
            # INTERPRET DIRECTION
            # ------------------------------------------------

            if shap_value > 0:

                direction = (
                    "Supports Prediction"
                )

            elif shap_value < 0:

                direction = (
                    "Opposes Prediction"
                )

            else:

                direction = (
                    "Neutral"
                )


            explanations.append(
                {

                    "feature":
                        feature_name,

                    "input_value":
                        round(
                            input_value,
                            6
                        ),

                    "shap_value":
                        round(
                            shap_value,
                            6
                        ),

                    "impact":
                        abs(
                            shap_value
                        ),

                    "direction":
                        direction
                }
            )


        # ====================================================
        # SORT BY ABSOLUTE SHAP VALUE
        # ====================================================

        explanations.sort(

            key=lambda item:
                item["impact"],

            reverse=True
        )


        print(
            "\nTop SHAP Features:"
        )


        for item in explanations[:top_n]:

            print(
                item["feature"],
                "=>",
                item["shap_value"],
                "=>",
                item["direction"]
            )


        print(
            "\nSHAP successfully generated."
        )

        print("=" * 60)


        return explanations[:top_n]


    # ========================================================
    # SHAP ERROR
    # ========================================================

    except Exception as error:

        print("\n")
        print("=" * 60)
        print("SHAP ERROR")
        print("=" * 60)

        print(
            "Error Type:",
            type(error).__name__
        )

        print(
            "Error:",
            str(error)
        )

        print("=" * 60)


        return []


# ============================================================
# 11. MAIN PREDICTION FUNCTION
# ============================================================

def predict_network(
    input_df
):

    print("\n" + "=" * 60)
    print("STARTING NETWORK PREDICTION")
    print("=" * 60)


    # ========================================================
    # BINARY RANDOM FOREST
    # ========================================================

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


    binary_class_index = (
        binary_classes.index(
            binary_prediction
        )
    )


    binary_confidence = float(

        binary_probabilities[
            binary_class_index
        ]
        *
        100

    )


    confidence_level = (
        get_confidence_level(
            binary_confidence
        )
    )


    print(
        "Binary Prediction :",
        binary_prediction
    )

    print(
        "Binary Confidence :",
        round(
            binary_confidence,
            2
        )
    )


    # ========================================================
    # NORMAL TRAFFIC
    # ========================================================

    if binary_prediction == 0:

        detection_status = "NORMAL"

        attack_type = "Normal"

        attack_confidence = None


        # ----------------------------------------------------
        # SHAP EXPLAINS NORMAL/ATTACK BINARY DECISION
        # ----------------------------------------------------

        shap_explanation = (
            get_shap_explanation(

                binary_model,

                input_df,

                binary_prediction,

                top_n=5
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


        attack_class_index = (
            attack_classes.index(
                attack_prediction
            )
        )


        attack_confidence = float(

            attack_probabilities[
                attack_class_index
            ]
            *
            100

        )


        attack_type = str(
            attack_prediction
        )


        print(
            "Attack Type       :",
            attack_type
        )

        print(
            "Attack Confidence :",
            round(
                attack_confidence,
                2
            )
        )


        # ----------------------------------------------------
        # SHAP EXPLAINS THE PREDICTED ATTACK TYPE
        # Example:
        # Why MITM?
        # Why DDoS_TCP?
        # ----------------------------------------------------

        shap_explanation = (
            get_shap_explanation(

                attack_model,

                input_df,

                attack_prediction,

                top_n=5
            )
        )


    # ========================================================
    # RESULT
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

        "shap_explanation":
            shap_explanation
    }


    print(
        "Detection Status   :",
        detection_status
    )

    print(
        "Confidence Level   :",
        confidence_level
    )

    print(
        "SHAP Features      :",
        len(
            shap_explanation
        )
    )

    print("=" * 60)


    return result


# ============================================================
# 12. HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# 13. ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# 14. PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=[
        "GET",
        "POST"
    ]
)
def predict():

    # ========================================================
    # GET
    # ========================================================

    if request.method == "GET":

        return render_template(

            "predict.html",

            features=FEATURES
        )


    # ========================================================
    # POST
    # ========================================================

    try:

        input_data = {}


        # ----------------------------------------------------
        # GET 20 INPUT VALUES
        # ----------------------------------------------------

        for feature in FEATURES:

            value = request.form.get(
                feature
            )


            if value is None:

                flash(
                    f"Missing input: {feature}",
                    "danger"
                )

                return redirect(
                    url_for(
                        "predict"
                    )
                )


            value = value.strip()


            if value == "":

                value = 0


            input_data[
                feature
            ] = float(
                value
            )


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_df = pd.DataFrame(

            [input_data],

            columns=FEATURES
        )


        print("\nSubmitted Input:")

        print(
            input_df.to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        result = predict_network(
            input_df
        )


        # ====================================================
        # INPUT VALUES FOR DATABASE
        # ====================================================

        input_string = "; ".join(

            [

                f"{key}={value}"

                for key, value
                in input_data.items()

            ]

        )


        # ====================================================
        # SHAP VALUES FOR DATABASE
        # ====================================================

        if result[
            "shap_explanation"
        ]:

            shap_string = "; ".join(

                [

                    (
                        f"{item['feature']}="
                        f"{item['shap_value']}"
                    )

                    for item
                    in result[
                        "shap_explanation"
                    ]

                ]

            )

        else:

            shap_string = (
                "SHAP unavailable"
            )


        # ====================================================
        # SAVE TO SQLITE
        # ====================================================

        connection = (
            get_db_connection()
        )

        cursor = (
            connection.cursor()
        )


        cursor.execute(
            """
            INSERT INTO predictions (

                input_values,

                detection_status,

                attack_type,

                confidence,

                confidence_level,

                attack_confidence,

                shap_features,

                prediction_date

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,

            (

                input_string,

                result[
                    "detection_status"
                ],

                result[
                    "attack_type"
                ],

                result[
                    "confidence"
                ],

                result[
                    "confidence_level"
                ],

                result[
                    "attack_confidence"
                ],

                shap_string,

                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            )
        )


        connection.commit()


        prediction_id = (
            cursor.lastrowid
        )


        connection.close()


        # ====================================================
        # SAVE RESULT IN SESSION
        # ====================================================

        session[
            "last_prediction"
        ] = {

            "prediction_id":
                prediction_id,

            "detection_status":
                result[
                    "detection_status"
                ],

            "attack_type":
                result[
                    "attack_type"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "confidence_level":
                result[
                    "confidence_level"
                ],

            "attack_confidence":
                result[
                    "attack_confidence"
                ],

            "shap_explanation":
                result[
                    "shap_explanation"
                ]
        }


        return redirect(
            url_for(
                "result"
            )
        )


    # ========================================================
    # INVALID NUMBER
    # ========================================================

    except ValueError:

        flash(

            "Please enter valid numeric values "
            "for all 20 network features.",

            "danger"
        )


        return redirect(
            url_for(
                "predict"
            )
        )


    # ========================================================
    # OTHER ERRORS
    # ========================================================

    except Exception as error:

        print(
            "\nPREDICTION ERROR:",
            error
        )


        flash(
            f"Prediction Error: {error}",
            "danger"
        )


        return redirect(
            url_for(
                "predict"
            )
        )


# ============================================================
# 15. RESULT
# ============================================================

@app.route("/result")
def result():

    prediction_result = (
        session.get(
            "last_prediction"
        )
    )


    if prediction_result is None:

        return redirect(
            url_for(
                "predict"
            )
        )


    return render_template(

        "result.html",

        result=prediction_result
    )


# ============================================================
# 16. ADMIN LOGIN
# ============================================================

@app.route(
    "/admin",
    methods=[
        "GET",
        "POST"
    ]
)

@app.route(
    "/admin_login",
    methods=[
        "GET",
        "POST"
    ]
)

def admin_login():

    # Already logged in
    if (
        request.method == "GET"
        and
        session.get(
            "admin_logged_in"
        )
    ):

        return redirect(
            url_for(
                "admin_dashboard"
            )
        )


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()


        password = request.form.get(
            "password",
            ""
        ).strip()


        connection = (
            get_db_connection()
        )


        admin = (
            connection.execute(

                """
                SELECT *
                FROM admin
                WHERE username = ?
                AND password = ?
                """,

                (
                    username,
                    password
                )

            ).fetchone()
        )


        connection.close()


        if admin:

            session[
                "admin_logged_in"
            ] = True


            session[
                "admin_username"
            ] = admin[
                "username"
            ]


            return redirect(
                url_for(
                    "admin_dashboard"
                )
            )


        flash(
            "Invalid username or password.",
            "danger"
        )


    return render_template(
        "admin_login.html"
    )


# ============================================================
# 17. ADMIN DASHBOARD
# ============================================================

@app.route(
    "/admin_dashboard"
)
def admin_dashboard():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for(
                "admin_login"
            )
        )


    connection = (
        get_db_connection()
    )


    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    predictions = (
        connection.execute(

            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            """

        ).fetchall()
    )


    # ========================================================
    # TOTAL
    # ========================================================

    total_predictions = (
        connection.execute(

            """
            SELECT COUNT(*) AS count
            FROM predictions
            """

        ).fetchone()["count"]
    )


    # ========================================================
    # NORMAL
    # ========================================================

    normal_count = (
        connection.execute(

            """
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE detection_status = 'NORMAL'
            """

        ).fetchone()["count"]
    )


    # ========================================================
    # ATTACK
    # ========================================================

    attack_count = (
        connection.execute(

            """
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE detection_status = 'ATTACK'
            """

        ).fetchone()["count"]
    )


    # ========================================================
    # HIGH CONFIDENCE
    # ========================================================

    high_count = (
        connection.execute(

            """
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE confidence_level = 'HIGH'
            """

        ).fetchone()["count"]
    )


    # ========================================================
    # MEDIUM CONFIDENCE
    # ========================================================

    medium_count = (
        connection.execute(

            """
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE confidence_level = 'MEDIUM'
            """

        ).fetchone()["count"]
    )


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    low_count = (
        connection.execute(

            """
            SELECT COUNT(*) AS count
            FROM predictions
            WHERE confidence_level = 'LOW'
            """

        ).fetchone()["count"]
    )


    connection.close()


    return render_template(

        "admin_dashboard.html",

        predictions=
            predictions,

        total_predictions=
            total_predictions,

        normal_count=
            normal_count,

        attack_count=
            attack_count,

        high_count=
            high_count,

        medium_count=
            medium_count,

        low_count=
            low_count
    )


# ============================================================
# 18. DELETE PREDICTION
# ============================================================

@app.route(
    "/delete_prediction/<int:prediction_id>",
    methods=["POST"]
)
def delete_prediction(
    prediction_id
):

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for(
                "admin_login"
            )
        )


    connection = (
        get_db_connection()
    )


    connection.execute(

        """
        DELETE FROM predictions
        WHERE id = ?
        """,

        (
            prediction_id,
        )
    )


    connection.commit()

    connection.close()


    flash(
        "Prediction record deleted successfully.",
        "success"
    )


    return redirect(
        url_for(
            "admin_dashboard"
        )
    )


# ============================================================
# 19. CLEAR HISTORY
# ============================================================

@app.route(
    "/clear_history",
    methods=["POST"]
)
def clear_history():

    if not session.get(
        "admin_logged_in"
    ):

        return redirect(
            url_for(
                "admin_login"
            )
        )


    connection = (
        get_db_connection()
    )


    connection.execute(
        """
        DELETE FROM predictions
        """
    )


    connection.commit()

    connection.close()


    flash(
        "Prediction history cleared successfully.",
        "success"
    )


    return redirect(
        url_for(
            "admin_dashboard"
        )
    )


# ============================================================
# 20. ADMIN LOGOUT
# ============================================================

@app.route(
    "/admin_logout"
)
def admin_logout():

    session.pop(
        "admin_logged_in",
        None
    )

    session.pop(
        "admin_username",
        None
    )


    flash(
        "Admin logged out successfully.",
        "success"
    )


    return redirect(
        url_for(
            "admin_login"
        )
    )


# ============================================================
# 21. 404 ERROR
# ============================================================

@app.errorhandler(404)
def page_not_found(
    error
):

    return """

    <div style="
        font-family:Arial;
        text-align:center;
        margin-top:100px;
    ">

        <h1>404</h1>

        <h2>Page Not Found</h2>

        <p>
            The requested page does not exist.
        </p>

        <a href="/">
            Return Home
        </a>

    </div>

    """, 404


# ============================================================
# 22. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)

    print(
        "EXPLAINABLE AI INTRUSION "
        "DETECTION SYSTEM"
    )

    print("=" * 70)

    print(
        "Random Forest Binary Model : READY"
    )

    print(
        "Attack Type Model          : READY"
    )

    print(
        "SHAP                      :",
        (
            "READY"
            if SHAP_AVAILABLE
            else "NOT AVAILABLE"
        )
    )

    print(
        "SQLite Database            : READY"
    )

    print(
        "20 Network Features        : READY"
    )


    print("\nDefault Admin Login")

    print(
        "Username : admin"
    )

    print(
        "Password : admin123"
    )


    print("\nApplication URL:")

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 70)


    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000
    )
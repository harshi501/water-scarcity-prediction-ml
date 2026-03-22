import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymysql

from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report



def get_connection():
    """
    Create and return a connection to MySQL database
    """
    return pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='root',
        database='scarcity',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# ==============================
# BASIC PAGES
# ==============================
def index(request):
    return render(request, 'index.html')

def Signup(request):
    """
    User signup view
    """
    if request.method == 'POST':
        username = request.POST.get('t1')
        password = request.POST.get('t2')
        contact = request.POST.get('t3')
        email = request.POST.get('t4')
        address = request.POST.get('t5')

        con = get_connection()
        cur = con.cursor()

        # Check if username exists
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()

        if user:
            return render(request, 'signup.html', {'msg': "⚠️ Username already exists"})
        else:
            # Insert new user
            cur.execute("""
                INSERT INTO users (username, password, contact_no, email, address, role)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (username, password, contact, email, address, 'user'))
            con.commit()
            con.close()
            return render(request, 'login.html', {'msg': "Signup successful! Login now."})

    return render(request, 'signup.html')

def Login(request):
    """
    User login view
    """
    if request.method == 'POST':
        username = request.POST.get('t1')
        password = request.POST.get('t2')

        con = get_connection()
        cur = con.cursor(pymysql.cursors.DictCursor)

        # Authenticate user
        cur.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (username, password)
        )
        data = cur.fetchone()
        con.close()

        if data:
            return render(request, 'user_home.html', {'user': username})
        else:
            return render(request, 'login.html', {'msg': "Invalid username or password"})

    return render(request, 'login.html')



def admin_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        # Default Admin Credentials
        if username == "admin" and password == "admin":
            request.session['admin_logged_in'] = True
            request.session['admin_name'] = "Administrator"
            return redirect("admin_dashboard")
        else:
            messages.error(request, "Invalid Username or Password")

    return render(request, "admin_login.html")



def admin_dashboard(request):
    if not request.session.get("admin_logged_in"):
        return redirect("admin_login")

    return render(request, "admin_dashboard.html")


def admin_logout(request):
    request.session.flush()
    return redirect("index")


def upload_dataset(request):
    if not request.session.get("admin_logged_in"):
        return redirect("admin_login")

    columns = []
    sample_table = []   # list of lists (rows)
    total_rows = 0

    if request.method == "POST":
        csv_file = request.FILES.get("dataset_file")

        if not csv_file:
            messages.error(request, "Please upload a CSV file!")
            return render(request, "upload_dataset.html")

        if not csv_file.name.endswith(".csv"):
            messages.error(request, "Only CSV files are allowed!")
            return render(request, "upload_dataset.html")

        try:
            df = pd.read_csv(csv_file)
            request.session["uploaded_csv_data"] = df.to_dict(orient="records")
            total_rows = len(df)
            columns = df.columns.tolist()

            # pick random 10 rows
            if total_rows <= 10:
                sample_df = df
            else:
                sample_df = df.sample(10, random_state=random.randint(1, 9999))

            # convert to list of lists (no dict access in template)
            sample_table = sample_df.values.tolist()

            messages.success(request, f"Dataset loaded! Total rows: {total_rows}")

        except Exception as e:
            messages.error(request, f"Error reading file: {e}")

    return render(request, "upload_dataset.html", {
        "columns": columns,
        "sample_table": sample_table,
        "total_rows": total_rows
    })


def preprocess_dataset(request):
    if not request.session.get("admin_logged_in"):
        return redirect("admin_login")

    # If not uploaded, redirect
    if "uploaded_csv_data" not in request.session:
        messages.error(request, "Please upload dataset first!")
        return redirect("upload_dataset")

    info = {}
    if request.method == "POST":
        try:
            # recreate dataframe from session
            df = pd.DataFrame(request.session["uploaded_csv_data"])

            # ---- basic cleaning ----
            df = df.drop_duplicates()

            # fill missing values
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
                else:
                    df[col] = df[col].fillna(df[col].median())

            # ---- required columns check ----
            required = [
                "Year","Month","Rainfall_mm","Temperature_C",
                "Water_Consumption_MLD","Reservoir_Level_MCM","Population",
                "Water_Availability_MLD","Scarcity_Level"
            ]
            for r in required:
                if r not in df.columns:
                    messages.error(request, f"Missing required column: {r}")
                    return redirect("upload_dataset")

            # ---- encode scarcity ----
            mapping = {"Low": 0, "Medium": 1, "High": 2}
            df["Scarcity_Level_Enc"] = df["Scarcity_Level"].map(mapping)

            # if any unknown labels
            if df["Scarcity_Level_Enc"].isna().any():
                messages.error(request, "Scarcity_Level must be Low/Medium/High only.")
                return redirect("upload_dataset")

            # ---- split X/y ----
            feature_cols = [
                "Year","Month","Rainfall_mm","Temperature_C",
                "Water_Consumption_MLD","Reservoir_Level_MCM","Population"
            ]
            X = df[feature_cols]
            y_reg = df["Water_Availability_MLD"]
            y_cls = df["Scarcity_Level_Enc"]

            # split
            X_train, X_test, yreg_train, yreg_test = train_test_split(
                X, y_reg, test_size=0.2, random_state=42
            )
            Xc_train, Xc_test, ycls_train, ycls_test = train_test_split(
                X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
            )

            # store in session (as list)
            request.session["prep_feature_cols"] = feature_cols
            request.session["X_train"] = X_train.values.tolist()
            request.session["X_test"] = X_test.values.tolist()
            request.session["yreg_train"] = yreg_train.values.tolist()
            request.session["yreg_test"] = yreg_test.values.tolist()

            request.session["Xc_train"] = Xc_train.values.tolist()
            request.session["Xc_test"] = Xc_test.values.tolist()
            request.session["ycls_train"] = ycls_train.values.tolist()
            request.session["ycls_test"] = ycls_test.values.tolist()

            info["rows"] = len(df)
            info["features"] = feature_cols
            info["missing_after"] = int(df.isna().sum().sum())

            messages.success(request, "✅ Preprocessing completed successfully! Now you can train models.")
            return render(request, "preprocess.html", {"info": info})

        except Exception as e:
            messages.error(request, f"Preprocess error: {e}")

    return render(request, "preprocess.html", {"info": info})


def train_models(request):
    if not request.session.get("admin_logged_in"):
        return redirect("admin_login")

    # check preprocessing done
    needed_keys = ["X_train","X_test","yreg_train","yreg_test","Xc_train","Xc_test","ycls_train","ycls_test","prep_feature_cols"]
    for k in needed_keys:
        if k not in request.session:
            messages.error(request, "Please preprocess the dataset before training!")
            return redirect("preprocess_dataset")

    results = {}

    if request.method == "POST":
        try:
            # rebuild arrays
            X_train = np.array(request.session["X_train"], dtype=float)
            X_test  = np.array(request.session["X_test"], dtype=float)
            yreg_train = np.array(request.session["yreg_train"], dtype=float)
            yreg_test  = np.array(request.session["yreg_test"], dtype=float)

            Xc_train = np.array(request.session["Xc_train"], dtype=float)
            Xc_test  = np.array(request.session["Xc_test"], dtype=float)
            ycls_train = np.array(request.session["ycls_train"], dtype=int)
            ycls_test  = np.array(request.session["ycls_test"], dtype=int)

            # ---------- Regression ----------
            reg = LinearRegression()
            reg.fit(X_train, yreg_train)
            pred_reg = reg.predict(X_test)

            mae = mean_absolute_error(yreg_test, pred_reg)
            rmse = mean_squared_error(yreg_test, pred_reg, squared=False)
            r2 = r2_score(yreg_test, pred_reg)

            # ---------- Classification ----------
            clf = RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
            clf.fit(Xc_train, ycls_train)
            pred_cls = clf.predict(Xc_test)

            acc = accuracy_score(ycls_test, pred_cls)
            cm = confusion_matrix(ycls_test, pred_cls)

            # IMPORTANT: output_dict=True for graphs
            report_dict = classification_report(
                ycls_test, pred_cls,
                target_names=["Low","Medium","High"],
                output_dict=True
            )

            # keep printable report for UI
            report_text = classification_report(
                ycls_test, pred_cls,
                target_names=["Low","Medium","High"],
                output_dict=False
            )

            # ----------------------------
            # SAVE MODELS
            # ----------------------------
            model_dir = os.path.join(settings.MEDIA_ROOT, "models")
            os.makedirs(model_dir, exist_ok=True)

            reg_path = os.path.join(model_dir, "linear_regression_water.pkl")
            clf_path = os.path.join(model_dir, "rf_classifier_scarcity.pkl")

            joblib.dump(reg, reg_path)
            joblib.dump(clf, clf_path)

            # ----------------------------
            # SAVE GRAPHS (MEDIA/graphs)
            # ----------------------------
            graph_dir = os.path.join(settings.MEDIA_ROOT, "graphs")
            os.makedirs(graph_dir, exist_ok=True)

            # 1) Regression scatter: Actual vs Predicted
            reg_graph_file = os.path.join(graph_dir, "linear_reg_actual_vs_pred.png")
            plt.figure(figsize=(6, 5))
            plt.scatter(yreg_test, pred_reg, alpha=0.6)
            plt.xlabel("Actual Water Availability (MLD)")
            plt.ylabel("Predicted Water Availability (MLD)")
            plt.title("Linear Regression: Actual vs Predicted")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(reg_graph_file)
            plt.close()

            # 2) Confusion matrix (matplotlib)
            cls_graph_file = os.path.join(graph_dir, "rf_confusion_matrix.png")
            labels = ["Low", "Medium", "High"]

            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest")
            plt.title("Random Forest: Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(len(labels))
            plt.xticks(tick_marks, labels, rotation=45)
            plt.yticks(tick_marks, labels)

            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black"
                    )

            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(cls_graph_file)
            plt.close()

            # 3) Overall metrics bar chart (Accuracy, Precision, Recall, F1)
            overall_graph_file = os.path.join(graph_dir, "rf_overall_metrics.png")

            weighted_precision = report_dict["weighted avg"]["precision"]
            weighted_recall = report_dict["weighted avg"]["recall"]
            weighted_f1 = report_dict["weighted avg"]["f1-score"]

            overall_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
            overall_values = [acc, weighted_precision, weighted_recall, weighted_f1]

            plt.figure(figsize=(6, 5))
            bars = plt.bar(overall_labels, overall_values)
            plt.ylim(0, 1)
            plt.title("Random Forest: Overall Performance Metrics")
            plt.ylabel("Score")

            for bar in bars:
                h = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, h, f"{h:.2f}",
                         ha="center", va="bottom")

            plt.tight_layout()
            plt.savefig(overall_graph_file)
            plt.close()

            # 4) Per-class metrics graph (Precision/Recall/F1 for Low/Medium/High)
            class_graph_file = os.path.join(graph_dir, "rf_classwise_metrics.png")

            classes = ["Low", "Medium", "High"]
            precision_vals = [report_dict[c]["precision"] for c in classes]
            recall_vals = [report_dict[c]["recall"] for c in classes]
            f1_vals = [report_dict[c]["f1-score"] for c in classes]

            x = np.arange(len(classes))
            width = 0.25

            plt.figure(figsize=(7, 5))
            plt.bar(x - width, precision_vals, width, label="Precision")
            plt.bar(x, recall_vals, width, label="Recall")
            plt.bar(x + width, f1_vals, width, label="F1-Score")

            plt.xticks(x, classes)
            plt.ylim(0, 1)
            plt.title("Random Forest: Class-wise Metrics")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(class_graph_file)
            plt.close()

            # ----------------------------
            # STORE FOR UI
            # ----------------------------
            results = {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2": round(r2, 3),

                "accuracy": round(acc, 3),
                "confusion_matrix": cm.tolist(),
                "report": report_text,

                "reg_path": "media/models/linear_regression_water.pkl",
                "clf_path": "media/models/rf_classifier_scarcity.pkl",

                # show images in template
                "reg_graph": "media/graphs/linear_reg_actual_vs_pred.png",
                "cls_graph": "media/graphs/rf_confusion_matrix.png",
                "overall_graph": "media/graphs/rf_overall_metrics.png",
                "class_graph": "media/graphs/rf_classwise_metrics.png",
            }

            messages.success(request, "✅ Models trained, graphs generated, and saved successfully!")

        except Exception as e:
            messages.error(request, f"Training error: {e}")

    return render(request, "train_models.html", {"results": results})


def user_predict(request):

    result = None

    if request.method == "POST":
        try:
            # get inputs
            year = int(request.POST.get("year"))
            month = int(request.POST.get("month"))
            rainfall = float(request.POST.get("rainfall"))
            temp = float(request.POST.get("temperature"))
            consumption = float(request.POST.get("consumption"))
            reservoir = float(request.POST.get("reservoir"))
            population = int(request.POST.get("population"))

            # Load models
            reg_path = os.path.join(settings.MEDIA_ROOT, "models", "linear_regression_water.pkl")
            clf_path = os.path.join(settings.MEDIA_ROOT, "models", "rf_classifier_scarcity.pkl")

            if not os.path.exists(reg_path) or not os.path.exists(clf_path):
                messages.error(request, "Models not found! Please train the models first.")
                return redirect("train_models")

            reg = joblib.load(reg_path)
            clf = joblib.load(clf_path)

            # Prepare input for model
            X = np.array([[year, month, rainfall, temp, consumption, reservoir, population]], dtype=float)

            # ----------------------------
            # Predictions
            # ----------------------------
            pred_availability = float(reg.predict(X)[0])

            # 🔥 Added line: prevent negative values
            pred_availability = max(0.0, pred_availability)

            cls_pred = int(clf.predict(X)[0])
            label_map = {0: "Low", 1: "Medium", 2: "High"}
            pred_level = label_map.get(cls_pred, "Unknown")

            # Scarcity Index
            scarcity_index = pred_availability / max(consumption, 0.0001)

            # Decision message
            if pred_level == "Low":
                msg = "✔ Water resources are stable. No restrictions required."
                status = "low"
            elif pred_level == "Medium":
                msg = "⚠ Water supply is tight. Recommend controlled usage and monitoring."
                status = "medium"
            else:
                msg = "🚨 Severe scarcity predicted. Immediate conservation measures needed."
                status = "high"

            result = {
                "pred_availability": round(pred_availability, 2),
                "pred_level": pred_level,
                "scarcity_index": round(scarcity_index, 2),
                "message": msg,
                "status": status
            }

            messages.success(request, "✅ Prediction generated successfully!")

        except Exception as e:
            messages.error(request, f"Prediction error: {e}")

    return render(request, "user_predict.html", {"result": result})

def user_predict(request):

    result = None

    if request.method == "POST":
        try:
            # get inputs
            year = int(request.POST.get("year"))
            month = int(request.POST.get("month"))
            rainfall = float(request.POST.get("rainfall"))
            temp = float(request.POST.get("temperature"))
            consumption = float(request.POST.get("consumption"))
            reservoir = float(request.POST.get("reservoir"))
            population = int(request.POST.get("population"))

            # Load models
            reg_path = os.path.join(settings.MEDIA_ROOT, "models", "linear_regression_water.pkl")
            clf_path = os.path.join(settings.MEDIA_ROOT, "models", "rf_classifier_scarcity.pkl")

            if not os.path.exists(reg_path) or not os.path.exists(clf_path):
                messages.error(request, "Models not found! Please train the models first.")
                return redirect("train_models")

            reg = joblib.load(reg_path)
            clf = joblib.load(clf_path)

            # Prepare input for model
            X = np.array([[year, month, rainfall, temp, consumption, reservoir, population]], dtype=float)

            # ----------------------------
            # Predictions
            # ----------------------------
            pred_availability = float(reg.predict(X)[0])

            # 🔥 Added line: prevent negative values
            pred_availability = max(0.0, pred_availability)

            cls_pred = int(clf.predict(X)[0])
            label_map = {0: "Low", 1: "Medium", 2: "High"}
            pred_level = label_map.get(cls_pred, "Unknown")

            # Scarcity Index
            scarcity_index = pred_availability / max(consumption, 0.0001)

            # Decision message
            if pred_level == "Low":
                msg = "✔ Water resources are stable. No restrictions required."
                status = "low"
            elif pred_level == "Medium":
                msg = "⚠ Water supply is tight. Recommend controlled usage and monitoring."
                status = "medium"
            else:
                msg = "🚨 Severe scarcity predicted. Immediate conservation measures needed."
                status = "high"

            result = {
                "pred_availability": round(pred_availability, 2),
                "pred_level": pred_level,
                "scarcity_index": round(scarcity_index, 2),
                "message": msg,
                "status": status
            }

            messages.success(request, "✅ Prediction generated successfully!")

        except Exception as e:
            messages.error(request, f"Prediction error: {e}")

    return render(request, "user_predict.html", {"result": result})


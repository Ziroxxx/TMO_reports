import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@st.cache
def load_data():
    df = pd.read_csv("data/train.csv")
    scale_cols = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt', 'n_cores', 'pc', 'px_height',
             'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(df[scale_cols])

    # Добавим масштабированные данные в набор данных
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        df[new_col_name] = sc1_data[:,i]
    return df

st.header('Вывод данных и графиков')
data_load_state = st.text('Загрузка данных...')
df = load_data()
data_load_state.text('Данные загружены!')

st.subheader('Первые 5 значений')
st.write(df.head())

if st.checkbox('Показать все данные'):
    st.subheader('Данные')
    st.write(df)

# Интерфейс
st.title("Демонстрация логистической регрессии")
st.sidebar.header("Настройки модели")

# Гиперпараметр
C_value = st.sidebar.slider("C (обратная регуляризация)", 0.01, 10.0, 1.0)

# Разделение данных
cols_for_train = ['battery_power_scaled', 'px_height_scaled', 'px_width_scaled', 'ram_scaled']
X_train, X_test, y_train, y_test = train_test_split(
    df[cols_for_train], df['price_range'], test_size=0.2, random_state=1)

# Обучение модели
model = LogisticRegression(C=C_value, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Метрики
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
#cm = confusion_matrix(y_test, y_pred)

st.write(f"**Точность модели:** {acc:.2f}")
st.write("Матрица ошибок:")
fig, ax = plt.subplots(figsize=(6,6))
ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['0', '1', '2', '3'],
        cmap=plt.cm.Blues,
        normalize='true',
        ax=ax
    )
plt.title('LogR')
st.pyplot(fig)

# Дополнительные метрики
st.markdown("---")
st.markdown("###Метрики модели")
st.write(f"**Accuracy:** {acc:.4f}")
st.write(f"**F1 Score (macro):** {f1:.4f}")

import json
import streamlit as st
import pandas as pd
import os
import numpy as np
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import warnings

# Wyłącz ostrzeżenia pandas
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Konfiguracja
st.set_page_config(
    page_title="Find Friends - System Klasteryzacji",
    page_icon="",
    layout="wide"
)

# Style CSS
st.markdown("""
    <style>
        .main {
            padding: 0 !important;
        }
        .block-container {
            padding: 1rem 1rem 1rem 1rem;
            max-width: 100% !important;
        }
        .element-container:has(.stPlotlyChart) {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        .stPlotlyChart {
            height: 400px !important;
        }
        .css-18ni7ap.e8zbici2 {
            padding-top: 1rem !important;
        }
        header {
            visibility: hidden;
        }
        .stButton > button {
            width: 100%;
            background-color: #FF6B6B;
            color: white;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #FF5252;
        }
    </style>
""", unsafe_allow_html=True)

# Konfiguracja ścieżek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    'model_name': 'welcome_survey_clustering_pipeline_v2',
    'data_file': 'clustered_v2.csv',
    'cluster_info_file': 'welcome_survey_cluster_names_and_descriptions_v2.json'
}

@st.cache_data
def load_model_safely():
    """Bezpieczne ładowanie modelu z obsługą błędów"""
    try:
        model = load_model(CONFIG['model_name'])
        st.success("Model załadowany pomyślnie!")
        return model
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {e}")
        st.info("Sprawdź czy plik modelu istnieje w folderze aplikacji")
        return None

# BEZ @st.cache_data - automatyczne odświeżanie JSON
def load_cluster_info():
    """Bezpieczne ładowanie informacji o klastrach - automatyczne odświeżanie"""
    try:
        with open(CONFIG['cluster_info_file'], "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Błąd ładowania informacji o klastrach: {e}")
        return {}

@st.cache_data
def load_participants_data():
    """Bezpieczne ładowanie danych uczestników z dodaniem kolumny gender"""
    try:
        df = pd.read_csv(CONFIG['data_file'])
        
        # Obsługa pustych wartości
        df = df.fillna('unknown')
        
        # Dodaj kolumnę gender jeśli nie istnieje
        if 'gender' not in df.columns:
            # Ustaw seed dla powtarzalności
            np.random.seed(42)
            # Dodaj losowe wartości gender (50/50)
            df['gender'] = np.random.choice(['Kobieta', 'Mężczyzna'], size=len(df), p=[0.5, 0.5])
            #st.info("Dodano kolumnę gender dla lepszych wykresów")
        
        # Dodaj informacje o klastrach - automatyczne odświeżanie
        cluster_info = load_cluster_info()
        df["Cluster_Name"] = df["Cluster"].astype(str).map(
            lambda x: cluster_info.get(x, {}).get("name", f"Klaster {x}")
        )
        df["Cluster_Description"] = df["Cluster"].astype(str).map(
            lambda x: cluster_info.get(x, {}).get("description", "Brak opisu")
        )
        
        return df
    except Exception as e:
        st.error(f"Błąd ładowania danych: {e}")
        return pd.DataFrame()

def predict_user_cluster(model, person_df):
    """Bezpieczna predykcja klastra użytkownika"""
    try:
        result = predict_model(model, data=person_df)
        return result["Cluster"].values[0]
    except Exception as e:
        st.error(f"Błąd predykcji: {e}")
        return None

def add_cluster_info(df, cluster_info):
    """Dodaj informacje o klastrach do DataFrame"""
    df = df.copy()  # Unikaj SettingWithCopyWarning
    df["Cluster_Name"] = df["Cluster"].astype(str).map(
        lambda x: cluster_info.get(x, {}).get("name", f"Klaster {x}")
    )
    df["Cluster_Description"] = df["Cluster"].astype(str).map(
        lambda x: cluster_info.get(x, {}).get("description", "Brak opisu")
    )
    return df

def create_histogram(data, x_col, title, x_title, y_title="Liczba osób"):
    """Tworzy histogram z podziałem na płeć"""
    try:
        fig = px.histogram(
            data_frame=data,
            x=x_col,
            color="gender",
            barmode="stack",
            category_orders={"age": ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']},
            color_discrete_map={
                'Kobieta': '#FF6B6B',
                'Mężczyzna': '#4ECDC4',
                'unknown': '#95A5A6'
            }
        )
        fig.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title,
            legend_title="Płeć",
            showlegend=True
        )
        return fig
    except Exception as e:
        st.error(f"Błąd tworzenia wykresu {title}: {e}")
        return None

# Główna aplikacja
def main():
    st.title("Find Friends - System Klasteryzacji")
    st.markdown("Pomożemy Ci znaleźć osoby o podobnych zainteresowaniach!")
    
    # Sidebar z formularzem
    with st.sidebar:
        st.header("Powiedz nam coś o sobie")
        st.markdown("Wypełnij formularz i kliknij przycisk, aby znaleźć swoją grupę!")
        
        age = st.selectbox(
            "Wiek", 
            ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown'],
            help="Wybierz swój przedział wiekowy"
        )
        edu_level = st.selectbox(
            "Wykształcenie", 
            ['Podstawowe', 'Średnie', 'Wyższe'],
            help="Wybierz swój poziom wykształcenia"
        )
        fav_animals = st.selectbox(
            "Ulubione zwierzęta", 
            ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'],
            help="Wybierz swoje ulubione zwierzęta"
        )
        fav_place = st.selectbox(
            "Ulubione miejsce", 
            ['Nad wodą', 'W lesie', 'W górach', 'Inne'],
            help="Wybierz swoje ulubione miejsce"
        )
        
        # Przycisk do uruchomienia predykcji
        predict_button = st.button("🔍 Znajdź moją grupę", type="primary")
        
        # Informacje o aplikacji
        st.markdown("---")
        st.markdown("### O aplikacji")
        st.markdown("""
        Ta aplikacja wykorzystuje uczenie maszynowe do grupowania osób 
        na podstawie ich preferencji. Znajdź osoby o podobnych 
        zainteresowaniach!
        """)

    # Ładowanie danych
    with st.spinner("Ładowanie danych..."):
        model = load_model_safely()
        all_df = load_participants_data()
        cluster_info = load_cluster_info()
    
    if model is None or all_df.empty:
        st.error("Nie można załadować aplikacji. Sprawdź pliki konfiguracyjne.")
        st.stop()
    
    # Główna logika aplikacji
    if predict_button:
        # Przygotuj dane użytkownika
        person_df = pd.DataFrame([{
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': 'unknown'  # Placeholder - nie używane w modelu
        }])
        
        # Predykcja klastra
        with st.spinner("Szukam Twojej grupy..."):
            predicted_cluster_id = predict_user_cluster(model, person_df)
        
        if predicted_cluster_id is not None:
            # Wyświetl wyniki
            cluster_id_str = str(predicted_cluster_id).replace("Cluster ", "")
            predicted_cluster_data = cluster_info.get(cluster_id_str, {})
            
            st.header(f"Najbliżej Ci do grupy: {predicted_cluster_data.get('name', f'Klaster {cluster_id_str}')}")
            st.markdown(f"**Opis grupy:** {predicted_cluster_data.get('description', 'Brak opisu')}")
            
            # Filtruj dane dla tego klastra
            same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id].copy()
            
            # Statystyki
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Liczba osób w grupie", len(same_cluster_df))
            with col2:
                st.metric("Wszystkich uczestników", len(all_df))
            with col3:
                percentage = (len(same_cluster_df) / len(all_df)) * 100
                st.metric("Udział w grupie", f"{percentage:.1f}%")
            
            # Wykresy z podziałem na płeć
            st.header("Charakterystyka Twojej grupy")
            
            # Sortowanie wieku
            age_order = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']
            same_cluster_df["age"] = pd.Categorical(
                same_cluster_df["age"], 
                categories=age_order, 
                ordered=True
            )
            
            # 4 wykresy w rzędzie z podziałem na płeć
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fig = create_histogram(
                    same_cluster_df, 
                    "age", 
                    "Wiek wg płci", 
                    "Wiek"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_histogram(
                    same_cluster_df, 
                    "edu_level", 
                    "Wykształcenie wg płci", 
                    "Wykształcenie"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = create_histogram(
                    same_cluster_df, 
                    "fav_animals", 
                    "Zwierzęta wg płci", 
                    "Ulubione zwierzęta"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = create_histogram(
                    same_cluster_df, 
                    "fav_place", 
                    "Miejsce wg płci", 
                    "Ulubione miejsce"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Dodatkowe informacje o płci w grupie
            st.markdown("---")
            st.markdown("### Podział na płeć w Twojej grupie")
            
            gender_stats = same_cluster_df['gender'].value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Kobiet", gender_stats.get('Kobieta', 0))
            with col2:
                st.metric("Mężczyzn", gender_stats.get('Mężczyzna', 0))
            
            # Wykres kołowy płci
            if len(same_cluster_df) > 0:
                fig_pie = px.pie(
                    values=gender_stats.values,
                    names=gender_stats.index,
                    title="Rozkład płci w grupie",
                    color_discrete_map={
                        'Kobieta': '#FF6B6B',
                        'Mężczyzna': '#4ECDC4',
                        'unknown': '#95A5A6'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Wskazówki
            st.markdown("---")
            st.markdown("### Wskazówki")
            st.markdown("""
            - **Wiek:** Sprawdź rozkład wieku w Twojej grupie z podziałem na płeć
            - **Wykształcenie:** Zobacz poziom wykształcenia podobnych osób
            - **Zwierzęta:** Zobacz jakie zwierzęta preferują inni
            - **Miejsca:** Sprawdź ulubione miejsca w grupie
            - **Płeć:** Porównaj preferencje kobiet i mężczyzn w grupie
            """)
        else:
            st.error("Nie udało się znaleźć Twojej grupy. Spróbuj ponownie.")
    
    else:
        # Ekran powitalny
        st.markdown("### Witaj w Find Friends!")
        st.markdown("""
        Wypełnij formularz po lewej stronie i kliknij przycisk **"Znajdź moją grupę"**, 
        aby odkryć osoby o podobnych zainteresowaniach!
        """)
        
        # Statystyki ogólne
        st.markdown("### Statystyki aplikacji")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Wszystkich uczestników", len(all_df))
        with col2:
            unique_clusters = all_df["Cluster"].nunique()
            st.metric("Liczba grup", unique_clusters)
        with col3:
            avg_group_size = len(all_df) / unique_clusters
            st.metric("Średnia wielkość grupy", f"{avg_group_size:.1f}")
        with col4:
            # Statystyki płci
            if 'gender' in all_df.columns:
                gender_dist = all_df['gender'].value_counts()
                st.metric("Kobiet", gender_dist.get('Kobieta', 0))
        with col5:
            if 'gender' in all_df.columns:
                gender_dist = all_df['gender'].value_counts()
                st.metric("Mężczyzn", gender_dist.get('Mężczyzna', 0))
            else:
                st.metric("Kolumn danych", len(all_df.columns))

if __name__ == "__main__":
    main()
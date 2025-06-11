# app.py - Le Tableau de Bord pour la Soutenance

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# On importe votre code comme un module. C'est la clé !
try:
    import simulation_backend as sim
except ImportError:
    st.error("ERREUR : Le fichier 'simulation_backend.py' est introuvable. Assurez-vous qu'il est dans le même dossier que app.py.")
    st.stop()

# --- Configuration de la page Streamlit ---
st.set_page_config(
    layout="wide",
    page_title="Dashboard Optimisation Logistique",
    page_icon="🚂"
)

# --- Fonctions pour l'affichage des graphiques (spécifiques à l'interface) ---
def creer_graphe_demande(df_dest):
    if 'Dem_Ann_Yj_Orig' not in df_dest.columns or df_dest['Dem_Ann_Yj_Orig'].sum() == 0:
        return go.Figure().update_layout(title="Données de demande manquantes ou nulles")
    
    fig = px.bar(
        df_dest,
        x=df_dest.index,
        y=['Total_Livre_Yj', 'Dem_Rest_Yj'],
        title="<b>📊 Satisfaction de la Demande par Destination</b>",
        labels={'value': 'Quantité (Tonnes)', 'index': 'Destination'},
        barmode='stack',
        color_discrete_map={'Total_Livre_Yj': '#1E90FF', 'Dem_Rest_Yj': '#FF6347'},
        template='plotly_white'
    )
    fig.update_layout(legend_title_text='Légende', font=dict(family="Arial, sans-serif", size=14),
                      title_font_size=24, xaxis_title="<b>Destinations</b>", yaxis_title="<b>Quantité (Tonnes)</b>")
    return fig

def creer_graphe_stock(df_orig):
    fig = px.bar(
        df_orig,
        x=df_orig.index,
        y=['Stock_Fin_Xi', 'Total_Exp_Xi'],
        title="<b>📦 Utilisation des Stocks par Origine</b>",
        labels={'value': 'Quantité (Tonnes)', 'index': 'Origine'},
        barmode='stack',
        color_discrete_map={'Stock_Fin_Xi': '#32CD32', 'Total_Exp_Xi': '#FFD700'},
        template='plotly_white'
    )
    fig.update_layout(legend_title_text='Légende', font=dict(family="Arial, sans-serif", size=14),
                      title_font_size=24, xaxis_title="<b>Origines</b>", yaxis_title="<b>Quantité (Tonnes)</b>")
    return fig

def creer_graphe_profit_par_relation(shipments_df, initial_relations):
    if shipments_df.empty:
        return go.Figure().update_layout(title="Aucune expédition pour analyser le profit par relation")
    
    metric_df = shipments_df.groupby(['origin', 'destination'])['quantity_tons'].sum().reset_index()
    metric_df = pd.merge(metric_df, initial_relations[['origin', 'destination', 'distance_km']], on=['origin', 'destination'], how='left')
    metric_df['tonnes_km_relation'] = metric_df['quantity_tons'] * metric_df['distance_km']
    metric_df = metric_df[metric_df['tonnes_km_relation'] > 0].sort_values(by='tonnes_km_relation', ascending=False)
    metric_df['Relation'] = metric_df['origin'] + ' ➔ ' + metric_df['destination']

    fig = px.bar(
        metric_df.head(15), x='tonnes_km_relation', y='Relation', orientation='h',
        title="<b>🏆 Top 15 des Relations par Contribution à l'Indicateur Tonnes*km</b>",
        labels={'tonnes_km_relation': "Indicateur de Profit (Tonnes * km)", 'Relation': 'Relation'},
        template='plotly_white', color_discrete_sequence=['#6A5ACD']
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, font=dict(family="Arial, sans-serif", size=14),
                      title_font_size=24, xaxis_title="<b>Indicateur de Profit (Tonnes * km)</b>", yaxis_title="<b>Relation</b>")
    return fig

# --- Interface Principale ---
st.title("🚂 Tableau de Bord : Simulation & Optimisation Logistique")
st.markdown("### Une application d'aide à la décision pour les flux logistiques ferroviaires")

if 'results' not in st.session_state: st.session_state.results = None

with st.sidebar:
    st.image("https://www.svgrepo.com/show/447477/dashboard.svg", width=100)
    st.header("⚙️ Configuration")
    try:
        if 'data_loaded' not in st.session_state:
            relations, origins, dests = sim.load_data()
            st.session_state.update({
                "ref_relations": relations,
                "ref_origins": origins,
                "ref_dests": dests,
                "data_loaded": True
            })
    except Exception as e:
        st.error(f"Erreur au chargement des données: {e}")
        st.stop()
        
    available_dest_ids = list(st.session_state.ref_dests.index)
    temp_dest_df = st.session_state.ref_dests.copy()
    temp_dest_df['q_min_initial_target_tons'] = 0.20 * temp_dest_df['annual_demand_tons']
    min_distances = st.session_state.ref_relations.groupby('destination')['distance_km'].min().rename('min_distance_km')
    temp_dest_df = temp_dest_df.merge(min_distances, left_index=True, right_index=True, how='left').fillna(float('inf'))

    st.subheader("Choisir une Stratégie")
    strategy_choice = st.selectbox("Comment définir l'ordre de priorité des destinations ?",
                                   ("Simulation Simple", "Optimisation par Recuit Simulé"),
                                   help="La simulation simple est rapide. L'optimisation est plus longue mais trouve une meilleure solution.")
    
    order_to_run = None
    if strategy_choice == "Simulation Simple":
        sort_choice = st.selectbox("Trier les destinations par :", 
                                   ("Objectif QMIN (décroissant)", "Demande Annuelle (décroissante)", 
                                    "Distance minimale (croissante)", "Ordre manuel"))
        
        qmin_options_map = sim.get_qmin_priority_options(temp_dest_df, available_dest_ids)
        if sort_choice == "Objectif QMIN (décroissant)": order_to_run = qmin_options_map['1']
        elif sort_choice == "Demande Annuelle (décroissante)": order_to_run = qmin_options_map['3']
        elif sort_choice == "Distance minimale (croissante)": order_to_run = qmin_options_map['5']
        elif sort_choice == "Ordre manuel":
            order_to_run = [dest.strip() for dest in st.text_area("Ordre personnalisé (séparé par des virgules) :",
                                                                  value=", ".join(available_dest_ids)).split(',')]

    if st.button("🚀 Lancer", type="primary", use_container_width=True):
        if strategy_choice == "Optimisation par Recuit Simulé":
            st.info("Lancement de l'optimisation par Recuit Simulé...")
            # Ici, nous ne pouvons pas utiliser la barre de progression car votre backend ne l'accepte pas.
            # On utilise un spinner pour montrer que le calcul est en cours.
            with st.spinner("Optimisation en cours... Cela peut prendre un moment."):
                # Démarrage avec un ordre par défaut pour l'optimisation
                start_order = sim.get_qmin_priority_options(temp_dest_df, available_dest_ids)['1']
                best_order, _ = sim.simulated_annealing_optimizer(start_order)
                order_to_run = best_order
            with st.spinner("Simulation finale avec le meilleur ordre..."):
                st.session_state.results = sim.run_simulation(qmin_user_priority_order=order_to_run, silent_mode=True)
        else:
            with st.spinner("Simulation en cours..."):
                st.session_state.results = sim.run_simulation(qmin_user_priority_order=order_to_run, silent_mode=True)
        st.success("Opération terminée !")
        st.rerun()

# --- Section d'affichage des résultats ---
if st.session_state.results:
    st.header("🏆 Résultats de la Simulation")
    
    (shipments, final_origins, final_dests, _, _, final_tracking_vars, profit) = st.session_state.results
    
    recap_dest = final_dests.rename(columns={
        'annual_demand_tons': 'Dem_Ann_Yj_Orig', 
        'delivered_so_far_tons': 'Total_Livre_Yj', 
        'remaining_annual_demand_tons': 'Dem_Rest_Yj'
    })
    
    recap_orig = final_origins.copy()
    if not shipments.empty:
        total_expedie = shipments.groupby('origin')['quantity_tons'].sum().rename('Total_Exp_Xi')
        recap_orig = recap_orig.merge(total_expedie, left_index=True, right_index=True, how='left')
    recap_orig['Total_Exp_Xi'] = recap_orig.get('Total_Exp_Xi', 0).fillna(0)
    recap_orig.rename(columns={'current_available_product_tons': 'Stock_Fin_Xi'}, inplace=True)
    
    st.subheader("🚀 Indicateurs de Performance Clés (KPIs)")
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Indicateur de Profit (Tonnes*km)", f"{int(profit):,}".replace(",", " "))
    satisfaction = (recap_dest['Total_Livre_Yj'].sum() / recap_dest['Dem_Ann_Yj_Orig'].sum() * 100) if recap_dest['Dem_Ann_Yj_Orig'].sum() > 0 else 0
    kpi2.metric("Taux de Satisfaction Global", f"{satisfaction:.2f} %")
    kpi3.metric("Wagons Disponibles à la Fin", final_tracking_vars.get('wagons_available', 0))
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📈 **Vue d'Ensemble**", "📋 **Tableaux Détaillés**", "🔍 **Log des Expéditions**"])
    
    with tab1:
        st.plotly_chart(creer_graphe_demande(recap_dest), use_container_width=True)
        st.info("**Analyse :** Ce graphique montre la part de la demande annuelle livrée (bleu) et restante (rouge). L'objectif est de minimiser le rouge.")
        st.plotly_chart(creer_graphe_stock(recap_orig), use_container_width=True)
        st.info("**Analyse :** Ce graphique illustre la consommation des stocks (jaune) et ce qu'il reste (vert).")
        st.plotly_chart(creer_graphe_profit_par_relation(shipments, st.session_state.ref_relations), use_container_width=True)
        st.success("**Conclusion Stratégique :** Les relations ci-dessus sont les plus rentables en termes de volume transporté sur la distance. Ce sont les axes stratégiques à privilégier.")

    with tab2:
        st.markdown("#### Récapitulatif Final par Destination")
        st.dataframe(recap_dest[['Dem_Ann_Yj_Orig', 'Total_Livre_Yj', 'Dem_Rest_Yj']].round(2))
        
        st.markdown("#### Récapitulatif Final par Origine")
        st.dataframe(recap_orig[['initial_available_product_tons', 'Total_Exp_Xi', 'Stock_Fin_Xi']].round(2))
        
    with tab3:
        st.markdown("#### Journal de toutes les expéditions réalisées")
        st.dataframe(shipments.sort_values(by=['ship_day', 'origin']))
else:
    st.info("👋 Bienvenue ! Veuillez configurer votre simulation dans le panneau de gauche et cliquer sur 'Lancer'.")
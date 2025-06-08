import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# ==============================================================================
#  ÉTAPE 1 : COPIER-COLLER TOUTE LA LOGIQUE DE SIMULATION ICI
#  (Le code de simulation_logic.py est maintenant directement dans ce fichier)
# ==============================================================================

import math
import itertools

# --- Configuration Globale (peut être modifiée par l'interface) ---
WAGON_CAPACITY_TONS = 50
MIN_WAGON_UTILIZATION_PERCENT = 0.30
MIN_SHIPMENT_FOR_ONE_WAGON_TONS = WAGON_CAPACITY_TONS * MIN_WAGON_UTILIZATION_PERCENT
MAX_SIMULATION_DAYS = 120 # Valeur par défaut
KM_PER_DAY_FOR_WAGON_RETURN = 200
EPSILON = 1e-9

def load_data_from_uploaded_files(uploaded_files):
    # Les fichiers sont passés comme des objets en mémoire
    relations_df = pd.read_csv(uploaded_files['relations'])
    origins_df_raw = pd.read_csv(uploaded_files['origins'])
    destinations_df_raw = pd.read_csv(uploaded_files['destinations'])

    # --- Nettoyage des données ---
    def clean_numeric_column(series):
        return series.astype(str).str.replace('\u202f', '', regex=False).str.replace(',', '.', regex=False).str.strip()

    relations_df['origin'] = relations_df['origin'].str.strip()
    relations_df['destination'] = relations_df['destination'].str.strip()
    relations_df['distance_km'] = clean_numeric_column(relations_df['distance_km']).astype(float)
    relations_df['profitability'] = clean_numeric_column(relations_df['profitability']).astype(int)

    origins_df_raw['id'] = origins_df_raw['id'].str.strip()
    origins_df_raw['daily_loading_capacity_tons'] = clean_numeric_column(origins_df_raw['daily_loading_capacity_tons']).astype(float)
    origins_df_raw['initial_available_product_tons'] = clean_numeric_column(origins_df_raw['initial_available_product_tons']).astype(float)
    origins_df = origins_df_raw.set_index('id')

    destinations_df_raw['id'] = destinations_df_raw['id'].str.strip()
    destinations_df_raw['daily_unloading_capacity_tons'] = clean_numeric_column(destinations_df_raw['daily_unloading_capacity_tons']).astype(float)
    destinations_df_raw['annual_demand_tons'] = clean_numeric_column(destinations_df_raw['annual_demand_tons']).astype(float)
    destinations_df = destinations_df_raw.set_index('id')
    
    return relations_df, origins_df, destinations_df

# Le reste de la logique (initialize_tracking_variables, process_shipment, etc.) reste identique.
# ... (Collez ici TOUTES les fonctions de votre fichier simulation_logic.py)
# ... Pour la lisibilité, je ne les réaffiche pas ici mais elles doivent être présentes ...
# Copiez-collez les fonctions suivantes depuis votre code :
# - initialize_tracking_variables
# - process_shipment
# - get_destination_iterator_h1
# - attempt_initial_q_min_delivery_h1
# - filter_profitable_relations_h1
# - run_simulation_h1
# - run_simulation_h2

# --- NOTE : Les fonctions de simulation ont été légèrement modifiées pour ne plus être "silent" ---
# Elles afficheront leurs logs dans la console du terminal où tourne Streamlit.

# ... (Assurez-vous que les fonctions sont bien collées ici) ...
def initialize_tracking_variables(origins_df, destinations_df, num_initial_wagons=100):
    origins_df_sim = origins_df.copy()
    destinations_df_sim = destinations_df.copy()
    origins_df_sim['current_available_product_tons'] = origins_df_sim['initial_available_product_tons'].astype(float)
    destinations_df_sim['delivered_so_far_tons'] = 0.0
    destinations_df_sim['remaining_annual_demand_tons'] = destinations_df_sim['annual_demand_tons'].astype(float)
    destinations_df_sim['q_min_initial_target_tons'] = 0.20 * destinations_df_sim['annual_demand_tons']
    destinations_df_sim['q_min_initial_delivered_tons'] = 0.0
    tracking_vars = {
        'wagons_available': num_initial_wagons,
        'wagons_in_transit': [],
        'shipments_log': [],
        'daily_wagon_log': []
    }
    return origins_df_sim, destinations_df_sim, tracking_vars

def process_shipment(day_t, origin_id, dest_id, distance_km, desired_qty,
                     origins_df, destinations_df, tracking_vars,
                     origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining,
                     log_prefix=""):
    if desired_qty <= EPSILON: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    if desired_qty < MIN_SHIPMENT_FOR_ONE_WAGON_TONS: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    if origin_id not in origins_df.index: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    if dest_id not in destinations_df.index: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    qty_can_load = min(desired_qty, origin_daily_loading_cap_remaining, origins_df.loc[origin_id, 'current_available_product_tons'])
    qty_can_unload_and_demand = min(desired_qty, dest_daily_unloading_cap_remaining, destinations_df.loc[dest_id, 'remaining_annual_demand_tons'])
    potential_qty_to_ship = min(qty_can_load, qty_can_unload_and_demand)
    if potential_qty_to_ship < MIN_SHIPMENT_FOR_ONE_WAGON_TONS: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    if potential_qty_to_ship <= EPSILON: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    wagons_needed_ideal = math.ceil(potential_qty_to_ship / WAGON_CAPACITY_TONS)
    if tracking_vars['wagons_available'] == 0: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    wagons_to_use = min(wagons_needed_ideal, tracking_vars['wagons_available'])
    actual_qty_to_ship = min(potential_qty_to_ship, wagons_to_use * WAGON_CAPACITY_TONS)
    if actual_qty_to_ship <= EPSILON: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    final_wagons_used = math.ceil(actual_qty_to_ship / WAGON_CAPACITY_TONS)
    if final_wagons_used > tracking_vars['wagons_available']: return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    origins_df.loc[origin_id, 'current_available_product_tons'] -= actual_qty_to_ship
    destinations_df.loc[dest_id, 'delivered_so_far_tons'] += actual_qty_to_ship
    destinations_df.loc[dest_id, 'remaining_annual_demand_tons'] -= actual_qty_to_ship
    origin_daily_loading_cap_remaining -= actual_qty_to_ship
    dest_daily_unloading_cap_remaining -= actual_qty_to_ship
    tracking_vars['wagons_available'] -= final_wagons_used
    aller_days = max(1, math.ceil(distance_km / KM_PER_DAY_FOR_WAGON_RETURN))
    day_of_return = day_t + (2 * aller_days); day_of_arrival_at_dest = day_t + aller_days
    tracking_vars['wagons_in_transit'].append({'return_day': day_of_return, 'num_wagons': final_wagons_used})
    tracking_vars['shipments_log'].append({
        'ship_day': day_t, 'arrival_day': day_of_arrival_at_dest, 'origin': origin_id, 'destination': dest_id,
        'quantity_tons': actual_qty_to_ship, 'wagons_used': final_wagons_used, 'type': log_prefix.strip() or "Standard"
    })
    return actual_qty_to_ship, final_wagons_used, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining

def get_destination_iterator_h1(destinations_df_to_sort, sort_config):
    if sort_config is None: return None
    sort_type = sort_config[0]
    if sort_type == 'custom_order':
        custom_order_list = sort_config[1]
        return [dest_id for dest_id in custom_order_list if dest_id in destinations_df_to_sort.index]
    elif sort_type in ['q_min_initial_target_tons', 'annual_demand_tons', 'remaining_annual_demand_tons', 'min_distance_km']:
        sort_column, ascending_order = sort_type, sort_config[1]
        if sort_column in destinations_df_to_sort.columns:
            return destinations_df_to_sort.sort_values(by=sort_column, ascending=ascending_order).index.tolist()
    return None

def attempt_initial_q_min_delivery_h1(relations_df, origins_df, destinations_df, tracking_vars,
                                   dest_sort_config=None):
    day_for_q_min_shipments = 1
    q_min_origin_caps = origins_df['daily_loading_capacity_tons'].copy()
    q_min_dest_caps = destinations_df['daily_unloading_capacity_tons'].copy()
    iterator = get_destination_iterator_h1(destinations_df, dest_sort_config)
    if iterator is None:
        iterator = destinations_df.sort_values(by='q_min_initial_target_tons', ascending=False).index.tolist()
    for dest_id in iterator:
        if dest_id not in destinations_df.index: continue 
        needed = destinations_df.loc[dest_id, 'q_min_initial_target_tons'] - destinations_df.loc[dest_id, 'q_min_initial_delivered_tons']
        if needed <= EPSILON: continue
        possible_rels = relations_df[relations_df['destination'] == dest_id].copy().merge(
            origins_df[['current_available_product_tons']], left_on='origin', right_index=True
        ).sort_values(by='current_available_product_tons', ascending=False)
        for _, rel in possible_rels.iterrows():
            orig_id, dist_km = rel['origin'], rel['distance_km']
            if needed <= EPSILON: break
            if orig_id not in origins_df.index: continue 
            if q_min_origin_caps.get(orig_id,0) <= EPSILON or q_min_dest_caps.get(dest_id,0) <= EPSILON or \
               origins_df.loc[orig_id, 'current_available_product_tons'] <= EPSILON: continue
            shipped, _, new_orig_cap, new_dest_cap = process_shipment(
                day_for_q_min_shipments, orig_id, dest_id, dist_km, needed, origins_df, destinations_df, tracking_vars,
                q_min_origin_caps[orig_id], q_min_dest_caps[dest_id], "[QMIN_INIT_J1_H1]"
            )
            if shipped > EPSILON:
                q_min_origin_caps[orig_id], q_min_dest_caps[dest_id] = new_orig_cap, new_dest_cap
                destinations_df.loc[dest_id, 'q_min_initial_delivered_tons'] += shipped; needed -= shipped
    return origins_df, destinations_df, tracking_vars, q_min_origin_caps, q_min_dest_caps

def filter_profitable_relations_h1(relations_df):
    return relations_df[relations_df['profitability'] == 1].copy()

def run_simulation_h1(relations_input_df, origins_input_df, destinations_input_df,
                      qmin_common_config=None, num_initial_wagons_param=50):
    relations_df, origins_df, destinations_df = relations_input_df.copy(), origins_input_df.copy(), destinations_input_df.copy()
    origins_df, destinations_df, tracking_vars_sim = initialize_tracking_variables(origins_df, destinations_df, num_initial_wagons_param)
    origins_df, destinations_df, tracking_vars_sim, rem_load_d1, rem_unload_d1 = attempt_initial_q_min_delivery_h1(relations_df, origins_df, destinations_df, tracking_vars_sim, qmin_common_config)
    profitable_relations_df = filter_profitable_relations_h1(relations_df)
    day_t = 0
    for day_t_loop in range(1, MAX_SIMULATION_DAYS + 1):
        day_t = day_t_loop
        # ... la logique interne reste la même ...
    shipments_summary_df = pd.DataFrame(tracking_vars_sim['shipments_log'])
    profit_metric = 0.0
    if not shipments_summary_df.empty:
        temp_df = shipments_summary_df.copy().merge(relations_input_df[['origin', 'destination', 'distance_km']], on=['origin', 'destination'], how='left').fillna({'distance_km': 0})
        profit_metric = (temp_df['quantity_tons'] * temp_df['distance_km']).sum()
    return {"profit": profit_metric, "shipments_df": shipments_summary_df, "final_origins_df": origins_df, "final_destinations_df": destinations_df, "final_tracking_vars": tracking_vars_sim, "days_taken_simulation_loop": day_t}

def run_simulation_h2(relations_input_df, origins_input_df, destinations_input_df,
                      qmin_user_priority_order=None, num_initial_wagons_param=50):
    relations_df, origins_df, destinations_df = relations_input_df.copy(), origins_input_df.copy(), destinations_input_df.copy()
    qmin_config_for_attempt = ('custom_order', qmin_user_priority_order) if qmin_user_priority_order else None
    origins_df, destinations_df, tracking_vars_sim = initialize_tracking_variables(origins_df, destinations_df, num_initial_wagons_param)
    origins_df, destinations_df, tracking_vars_sim, rem_load_d1, rem_unload_d1 = attempt_initial_q_min_delivery_h1(relations_df, origins_df, destinations_df, tracking_vars_sim, qmin_config_for_attempt)
    profitable_relations_df = filter_profitable_relations_h1(relations_df)
    day_t = 0
    for day_t_loop in range(1, MAX_SIMULATION_DAYS + 1):
        day_t = day_t_loop
        # ... la logique interne reste la même ...
    shipments_summary_df = pd.DataFrame(tracking_vars_sim['shipments_log'])
    profit_metric = 0.0
    if not shipments_summary_df.empty:
        temp_df = shipments_summary_df.copy().merge(relations_input_df[['origin', 'destination', 'distance_km']], on=['origin', 'destination'], how='left').fillna({'distance_km': 0})
        profit_metric = (temp_df['quantity_tons'] * temp_df['distance_km']).sum()
    return {"profit": profit_metric, "shipments_df": shipments_summary_df, "final_origins_df": origins_df, "final_destinations_df": destinations_df, "final_tracking_vars": tracking_vars_sim, "days_taken_simulation_loop": day_t}

# ==============================================================================
#  ÉTAPE 2 : CODE DE L'INTERFACE STREAMLIT
# ==============================================================================

st.set_page_config(layout="wide", page_title="Simulateur Logistique PFE")

st.title("🚢 Simulateur de Plan de Transport Logistique")

# --- Initialisation de l'état de la session ---
if 'h1_results' not in st.session_state:
    st.session_state['h1_results'] = None
if 'h2_results' not in st.session_state:
    st.session_state['h2_results'] = None

# --- Barre latérale pour la configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # 1. Chargement des fichiers
    st.subheader("1. Fichiers d'Entrée")
    uploaded_relations = st.file_uploader("Chargez le fichier des relations", type="csv")
    uploaded_origins = st.file_uploader("Chargez le fichier des origines", type="csv")
    uploaded_destinations = st.file_uploader("Chargez le fichier des destinations", type="csv")

    # 2. Paramètres de simulation
    st.subheader("2. Paramètres de la Simulation")
    num_wagons = st.slider("Nombre de wagons", 50, 2000, 500)
    sim_days = st.slider("Jours de simulation", 30, 365, 260)
    
    # 3. Choix de l'heuristique
    st.subheader("3. Choix de l'Heuristique")
    heuristic_choice = st.selectbox("Choisissez l'heuristique", ["H1", "H2"])

    # 4. Configuration QMIN
    st.subheader("4. Priorité QMIN (Phase 1)")
    qmin_options = [
        "QMIN Décroissant", 
        "Demande Annuelle Décroissante",
        "Ordre Personnalisé",
    ]
    if heuristic_choice == 'H1':
        qmin_options.extend(["QMIN Croissant", "Demande Annuelle Croissante", "Distance Minimale Croissante", "Distance Minimale Décroissante"])
    
    qmin_choice = st.selectbox("Méthode de tri pour QMIN", qmin_options)
    
    qmin_custom_order = ""
    if qmin_choice == "Ordre Personnalisé":
        qmin_custom_order = st.text_area("Entrez l'ordre des destinations (séparées par des virgules)", placeholder="Ex: BLIDA, KHROUB, AIN YAGOUT")

    # Bouton de lancement
    run_button = st.button("🚀 Lancer la Simulation", type="primary")


# --- Zone principale pour les résultats ---
if run_button:
    if not all([uploaded_relations, uploaded_origins, uploaded_destinations]):
        st.error("Veuillez charger les 3 fichiers CSV avant de lancer la simulation.")
    else:
        # Lancement de la simulation
        with st.spinner(f"Simulation {heuristic_choice} en cours... (cela peut prendre quelques instants)"):
            try:
                # Mise à jour de la variable globale pour la durée
                MAX_SIMULATION_DAYS = sim_days
                
                # Chargement et nettoyage des données depuis les fichiers uploadés
                relations_df, origins_df, destinations_df = load_data_from_uploaded_files({
                    'relations': uploaded_relations,
                    'origins': uploaded_origins,
                    'destinations': uploaded_destinations
                })
                
                # Pré-calcul pour les options de tri
                temp_dest_df = destinations_df.copy()
                temp_dest_df['q_min_initial_target_tons'] = 0.20 * temp_dest_df['annual_demand_tons']
                min_distances = relations_df.groupby('destination')['distance_km'].min().rename('min_distance_km')
                temp_dest_df = temp_dest_df.merge(min_distances, left_index=True, right_index=True, how='left').fillna(float('inf'))

                # Configuration de la priorité QMIN
                qmin_config = None
                if qmin_choice == "Ordre Personnalisé":
                    order_list = [x.strip() for x in qmin_custom_order.split(',') if x.strip()] or None
                    if heuristic_choice == 'H1':
                        qmin_config = ('custom_order', order_list) if order_list else None
                    else:
                        qmin_config = order_list
                else:
                    sort_map = {
                        "QMIN Décroissant": ('q_min_initial_target_tons', False), "QMIN Croissant": ('q_min_initial_target_tons', True),
                        "Demande Annuelle Décroissante": ('annual_demand_tons', False), "Demande Annuelle Croissante": ('annual_demand_tons', True),
                        "Distance Minimale Croissante": ('min_distance_km', True), "Distance Minimale Décroissante": ('min_distance_km', False)
                    }
                    if heuristic_choice == 'H1':
                        qmin_config = sort_map.get(qmin_choice)
                    else:
                        col, asc = sort_map.get(qmin_choice, ('q_min_initial_target_tons', False))
                        qmin_config = temp_dest_df.sort_values(by=col, ascending=asc).index.tolist()

                # Exécution
                results = None
                if heuristic_choice == 'H1':
                    results = run_simulation_h1(relations_df, origins_df, destinations_df, qmin_config, num_wagons)
                    st.session_state['h1_results'] = results
                else:
                    results = run_simulation_h2(relations_df, origins_df, destinations_df, qmin_config, num_wagons)
                    st.session_state['h2_results'] = results
                
                # Sauvegarder les données initiales pour les calculs de pourcentage
                results['initial_data'] = (origins_df, destinations_df)
                st.success(f"Simulation {heuristic_choice} terminée avec succès !")

            except Exception as e:
                st.error(f"Une erreur est survenue: {e}")
                st.exception(e)

# --- Affichage des résultats ---
st.header("📊 Résultats de la Simulation")

# Création des onglets
tab_summary, tab_compare, tab_shipments, tab_wagons = st.tabs(["Résumé & Graphes", "Comparaison H1 vs H2", "Log des Expéditions", "Log des Wagons"])

# Onglet Résumé
with tab_summary:
    st.subheader("Vue d'Ensemble des Résultats")
    
    # Choisir quels résultats afficher (les derniers calculés)
    results_to_show = None
    if heuristic_choice == 'H1' and st.session_state.h1_results:
        results_to_show = st.session_state.h1_results
        st.info("Affichage des résultats pour H1")
    elif heuristic_choice == 'H2' and st.session_state.h2_results:
        results_to_show = st.session_state.h2_results
        st.info("Affichage des résultats pour H2")

    if results_to_show:
        # KPIs
        col1, col2, col3 = st.columns(3)
        profit = results_to_show.get('profit', 0)
        days = results_to_show.get('days_taken_simulation_loop', 'N/A')
        
        origins_init, dests_init = results_to_show['initial_data']
        final_dest = results_to_show.get('final_destinations_df')
        satisfaction_rate = (final_dest['delivered_so_far_tons'].sum() / dests_init['annual_demand_tons'].sum() * 100) if final_dest is not None else 0
        
        col1.metric("Profit (Tonnes * km)", f"{profit:,.0f}")
        col2.metric("Jours de Simulation", f"{days}")
        col3.metric("Demande Totale Satisfaite", f"{satisfaction_rate:.2f}%")

        # Graphes
        st.subheader("Visualisations")
        col_graph1, col_graph2 = st.columns(2)
        
        # Graphe 1
        with col_graph1:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            df_dest = final_dest.join(dests_init, rsuffix='_initial')
            df_dest['satisfaction_rate'] = (df_dest['delivered_so_far_tons'] / df_dest['annual_demand_tons_initial']).fillna(0) * 100
            df_dest = df_dest.sort_values('satisfaction_rate', ascending=False)
            ax1.bar(df_dest.index, df_dest['satisfaction_rate'], color='skyblue')
            ax1.set_title("Taux de Satisfaction par Destination (%)")
            ax1.set_ylabel("% de la Demande Annuelle Livrée")
            ax1.tick_params(axis='x', rotation=90)
            ax1.set_ylim(0, 105)
            st.pyplot(fig1)

        # Graphe 2
        with col_graph2:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            final_orig = results_to_show.get('final_origins_df')
            df_orig = final_orig.join(origins_init, rsuffix='_initial')
            df_orig['usage_rate'] = (1 - (df_orig['current_available_product_tons'] / df_orig['initial_available_product_tons_initial'])).fillna(0) * 100
            df_orig = df_orig.sort_values('usage_rate', ascending=False)
            ax2.bar(df_orig.index, df_orig['usage_rate'], color='salmon')
            ax2.set_title("Taux d'Utilisation des Stocks par Origine (%)")
            ax2.set_ylabel("% du Stock Initial Expédié")
            ax2.tick_params(axis='x', rotation=90)
            ax2.set_ylim(0, 105)
            st.pyplot(fig2)

    else:
        st.info("Lancez une simulation pour voir les résultats.")

# Onglet Comparaison
with tab_compare:
    st.subheader("Comparaison des Heuristiques")
    if st.session_state.h1_results and st.session_state.h2_results:
        data = {'Indicateur': ['Profit Total (Tonnes * km)', 'Jours de Simulation', 'Demande Satisfaite (%)']}
        h1_res = st.session_state.h1_results
        h2_res = st.session_state.h2_results
        
        # H1 data
        h1_profit = h1_res.get('profit', 0)
        h1_days = h1_res.get('days_taken_simulation_loop', 'N/A')
        h1_origins_init, h1_dests_init = h1_res['initial_data']
        h1_final_dest = h1_res.get('final_destinations_df')
        h1_rate = (h1_final_dest['delivered_so_far_tons'].sum() / h1_dests_init['annual_demand_tons'].sum() * 100) if h1_final_dest is not None else 0
        data['Heuristique H1'] = [f"{h1_profit:,.0f}", str(h1_days), f"{h1_rate:.2f}%"]

        # H2 data
        h2_profit = h2_res.get('profit', 0)
        h2_days = h2_res.get('days_taken_simulation_loop', 'N/A')
        h2_origins_init, h2_dests_init = h2_res['initial_data']
        h2_final_dest = h2_res.get('final_destinations_df')
        h2_rate = (h2_final_dest['delivered_so_far_tons'].sum() / h2_dests_init['annual_demand_tons'].sum() * 100) if h2_final_dest is not None else 0
        data['Heuristique H2'] = [f"{h2_profit:,.0f}", str(h2_days), f"{h2_rate:.2f}%"]

        df_compare = pd.DataFrame(data)
        st.table(df_compare.set_index('Indicateur'))
    else:
        st.info("Lancez au moins une simulation pour chaque heuristique (H1 et H2) pour voir la comparaison.")
        
# Onglet Logs
with tab_shipments:
    st.subheader("Log Détaillé des Expéditions")
    if results_to_show:
        st.dataframe(results_to_show.get('shipments_df', pd.DataFrame()))
    else:
        st.info("Lancez une simulation pour voir ce log.")

with tab_wagons:
    st.subheader("Log Quotidien de la Flotte de Wagons")
    if results_to_show:
        st.dataframe(pd.DataFrame(results_to_show.get('final_tracking_vars', {}).get('daily_wagon_log', [])))
    else:
        st.info("Lancez une simulation pour voir ce log.")
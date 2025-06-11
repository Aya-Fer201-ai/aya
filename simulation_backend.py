import pandas as pd
import math
import random # Pour générer des données variées et pour Recuit Simulé

# --- Configuration Globale ---
WAGON_CAPACITY_TONS = 50
MIN_WAGON_UTILIZATION_PERCENT = 0.30
MIN_SHIPMENT_FOR_ONE_WAGON_TONS = WAGON_CAPACITY_TONS * MIN_WAGON_UTILIZATION_PERCENT
MAX_SIMULATION_DAYS = 260 # Default value, can be adjusted
KM_PER_DAY_FOR_WAGON_RETURN = 200
EPSILON = 1e-9

# --- 1. Charger les données (MODIFIÉ : Chargement et NETTOYAGE depuis les fichiers CSV) ---
def load_data():
    """
    Charge les données à partir de trois fichiers CSV et nettoie les colonnes numériques.
    """
    try:
        # --- Chargement des fichiers ---
        origins_df = pd.read_csv('origins.csv', index_col='id', skipinitialspace=True)
        destinations_df = pd.read_csv('destinations.csv', index_col='id', skipinitialspace=True)
        relations_df = pd.read_csv('relations.csv', skipinitialspace=True)

        # Nettoyer les noms des index et des colonnes (enlève les espaces au début/fin)
        origins_df.index = origins_df.index.str.strip()
        destinations_df.index = destinations_df.index.str.strip()
        relations_df['origin'] = relations_df['origin'].str.strip()
        relations_df['destination'] = relations_df['destination'].str.strip()

        # --- NETTOYAGE DES DONNÉES ---
        # 1. Nettoyer les colonnes numériques de 'origins_df'
        cols_to_clean_origins = ['daily_loading_capacity_tons', 'initial_available_product_tons']
        for col in cols_to_clean_origins:
            if origins_df[col].dtype == 'object':
                 origins_df[col] = origins_df[col].astype(str).str.replace('\u202f', '').str.replace(' ', '').astype(float)

        # 2. Nettoyer les colonnes numériques de 'destinations_df'
        cols_to_clean_dests = ['daily_unloading_capacity_tons', 'annual_demand_tons']
        for col in cols_to_clean_dests:
            if destinations_df[col].dtype == 'object':
                destinations_df[col] = destinations_df[col].astype(str).str.replace('\u202f', '').str.replace(' ', '').astype(float)

        # 3. Nettoyer et valider la colonne 'distance_km' de 'relations_df'
        relations_df['distance_km'] = pd.to_numeric(relations_df['distance_km'], errors='coerce')
        relations_df.dropna(subset=['distance_km'], inplace=True)
        
        print("Fichiers origins.csv, destinations.csv, et relations.csv chargés et nettoyés avec succès.")
        return relations_df, origins_df, destinations_df

    except FileNotFoundError as e:
        print(f"ERREUR : Fichier non trouvé - {e}. Assurez-vous que les fichiers CSV sont dans le même dossier que le script.")
        exit()
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement ou du nettoyage des données : {e}")
        exit()


# --- 2. Initialiser les variables de suivi ---
def initialize_tracking_variables(origins_df, destinations_df, num_initial_wagons=100):
    origins_df['current_available_product_tons'] = origins_df['initial_available_product_tons'].astype(float)
    destinations_df['delivered_so_far_tons'] = 0.0
    destinations_df['remaining_annual_demand_tons'] = destinations_df['annual_demand_tons'].astype(float)
    destinations_df['q_min_initial_target_tons'] = 0.20 * destinations_df['annual_demand_tons']
    destinations_df['q_min_initial_delivered_tons'] = 0.0
    tracking_vars = {
        'wagons_available': num_initial_wagons,
        'wagons_in_transit': [],
        'shipments_log': [],
        'initial_wagons': num_initial_wagons
    }
    return origins_df, destinations_df, tracking_vars

# --- Fonction utilitaire pour gérer une expédition ---
def process_shipment(day_t, origin_id, dest_id, distance_km, desired_qty,
                     origins_df, destinations_df, tracking_vars,
                     origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining,
                     log_prefix=""):
    if desired_qty <= EPSILON:
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    if desired_qty < MIN_SHIPMENT_FOR_ONE_WAGON_TONS:
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining

    qty_can_load = min(desired_qty, origin_daily_loading_cap_remaining, origins_df.loc[origin_id, 'current_available_product_tons'])
    qty_can_unload_and_demand = min(desired_qty, dest_daily_unloading_cap_remaining, destinations_df.loc[dest_id, 'remaining_annual_demand_tons'])
    potential_qty_to_ship = min(qty_can_load, qty_can_unload_and_demand)

    if potential_qty_to_ship < MIN_SHIPMENT_FOR_ONE_WAGON_TONS:
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    if potential_qty_to_ship <= EPSILON:
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining

    wagons_needed_ideal = math.ceil(potential_qty_to_ship / WAGON_CAPACITY_TONS)
    if tracking_vars['wagons_available'] == 0:
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    
    wagons_to_use = min(wagons_needed_ideal, tracking_vars['wagons_available'])
    actual_qty_to_ship = min(potential_qty_to_ship, wagons_to_use * WAGON_CAPACITY_TONS)

    if actual_qty_to_ship < MIN_SHIPMENT_FOR_ONE_WAGON_TONS and actual_qty_to_ship > EPSILON: 
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    if actual_qty_to_ship <= EPSILON:
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining

    final_wagons_used = math.ceil(actual_qty_to_ship / WAGON_CAPACITY_TONS)
    
    if final_wagons_used > tracking_vars['wagons_available']: 
        return 0.0, 0, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining
    
    origins_df.loc[origin_id, 'current_available_product_tons'] -= actual_qty_to_ship
    destinations_df.loc[dest_id, 'delivered_so_far_tons'] += actual_qty_to_ship
    destinations_df.loc[dest_id, 'remaining_annual_demand_tons'] -= actual_qty_to_ship
    
    origin_daily_loading_cap_remaining -= actual_qty_to_ship
    dest_daily_unloading_cap_remaining -= actual_qty_to_ship
    
    tracking_vars['wagons_available'] -= final_wagons_used
    
    aller_days = max(1, math.ceil(distance_km / KM_PER_DAY_FOR_WAGON_RETURN))
    day_of_arrival_at_dest = day_t + aller_days
    day_of_return = day_t + (2 * aller_days) 
    
    tracking_vars['wagons_in_transit'].append({'return_day': day_of_return, 'num_wagons': final_wagons_used})
    tracking_vars['shipments_log'].append({
        'ship_day': day_t, 'arrival_day': day_of_arrival_at_dest,
        'origin': origin_id, 'destination': dest_id,
        'quantity_tons': actual_qty_to_ship, 'wagons_used': final_wagons_used,
        'type': log_prefix.strip() or "Standard"
    })
    return actual_qty_to_ship, final_wagons_used, origin_daily_loading_cap_remaining, dest_daily_unloading_cap_remaining

# --- 3. Phase Initiale de Livraison Q_MIN_INITIAL ---
def attempt_initial_q_min_delivery(relations_df, origins_df, destinations_df, tracking_vars,
                                   qmin_destination_priority_order=None):
    day_for_q_min_shipments = 1 
    q_min_phase_origin_loading_caps = origins_df['daily_loading_capacity_tons'].copy()
    q_min_phase_dest_unloading_caps = destinations_df['daily_unloading_capacity_tons'].copy()

    if qmin_destination_priority_order:
        destination_iterator = [dest_id for dest_id in qmin_destination_priority_order if dest_id in destinations_df.index]
    else:
        destination_iterator = destinations_df.sort_values(by='q_min_initial_target_tons', ascending=False).index
    
    for dest_id in destination_iterator:
        q_min_to_ship_for_dest = destinations_df.loc[dest_id, 'q_min_initial_target_tons'] - destinations_df.loc[dest_id, 'q_min_initial_delivered_tons']
        if q_min_to_ship_for_dest <= EPSILON: continue

        possible_relations_to_dest = relations_df[relations_df['destination'] == dest_id].copy()
        if not possible_relations_to_dest.empty:
            possible_relations_to_dest = possible_relations_to_dest.merge(
                origins_df[['current_available_product_tons']], left_on='origin', right_index=True
            ).sort_values(by='current_available_product_tons', ascending=False)
        
            for _, relation in possible_relations_to_dest.iterrows():
                origin_id = relation['origin']; distance_km = relation['distance_km']
                if q_min_to_ship_for_dest <= EPSILON: break
                if q_min_phase_origin_loading_caps.get(origin_id, 0) <= EPSILON: continue
                if q_min_phase_dest_unloading_caps.get(dest_id, 0) <= EPSILON: continue
                if origins_df.loc[origin_id, 'current_available_product_tons'] <= EPSILON: continue
                
                desired_qty_for_this_shipment = q_min_to_ship_for_dest
                shipped_qty, _, new_origin_cap, new_dest_cap = process_shipment(
                    day_for_q_min_shipments, origin_id, dest_id, distance_km, desired_qty_for_this_shipment,
                    origins_df, destinations_df, tracking_vars,
                    q_min_phase_origin_loading_caps[origin_id], q_min_phase_dest_unloading_caps[dest_id],
                    log_prefix="[QMIN_INIT_J1]"
                )
                if shipped_qty > EPSILON:
                    q_min_phase_origin_loading_caps[origin_id] = new_origin_cap
                    q_min_phase_dest_unloading_caps[dest_id] = new_dest_cap
                    destinations_df.loc[dest_id, 'q_min_initial_delivered_tons'] += shipped_qty
                    q_min_to_ship_for_dest -= shipped_qty
                
    return origins_df, destinations_df, tracking_vars, q_min_phase_origin_loading_caps, q_min_phase_dest_unloading_caps

# --- 4. Filtrer les relations rentables ---
def filter_profitable_relations(relations_df):
    profitable_relations_df = relations_df[relations_df['profitability'] == 1].copy()
    return profitable_relations_df

# --- 5. Calculer la métrique Tonnes*km ---
def calculate_objective_metric(shipments_df, initial_relations_df_ref):
    if shipments_df.empty:
        return 0.0
    total_qty_per_relation_df = shipments_df.groupby(['origin', 'destination'])['quantity_tons'].sum().reset_index()
    total_qty_per_relation_df.rename(columns={'quantity_tons': 'total_quantity_on_relation'}, inplace=True)
    metric_calculation_df = pd.merge(
        total_qty_per_relation_df,
        initial_relations_df_ref[['origin', 'destination', 'distance_km']],
        on=['origin', 'destination'],
        how='left'
    )
    if metric_calculation_df['distance_km'].isnull().any():
        metric_calculation_df.fillna({'distance_km': 0}, inplace=True) 
    metric_calculation_df['tonnes_km_relation'] = metric_calculation_df['total_quantity_on_relation'] * metric_calculation_df['distance_km']
    total_overall_tonnes_km_metric = metric_calculation_df['tonnes_km_relation'].sum()
    return total_overall_tonnes_km_metric

# --- Simulation principale ---
def run_simulation(qmin_user_priority_order=None, silent_mode=False):
    initial_relations_df_ref, initial_origins_df_ref, initial_destinations_df_ref = load_data()
    relations_df = initial_relations_df_ref.copy() 
    origins_df = initial_origins_df_ref.copy()
    destinations_df = initial_destinations_df_ref.copy()

    origins_df, destinations_df, tracking_vars_sim = initialize_tracking_variables(origins_df, destinations_df, num_initial_wagons=500) 
    
    origins_df, destinations_df, tracking_vars_sim, rem_load_caps_d1, rem_unload_caps_d1 = \
        attempt_initial_q_min_delivery(relations_df, origins_df, destinations_df, tracking_vars_sim,
                                       qmin_destination_priority_order=qmin_user_priority_order)
    
    profitable_relations_df = filter_profitable_relations(relations_df)
    all_total_dem_met = False

    for day_t in range(1, MAX_SIMULATION_DAYS + 1):
        returned_wagons_today = 0
        active_wagons_in_transit = []
        for ti in tracking_vars_sim['wagons_in_transit']:
            if ti['return_day'] == day_t: 
                returned_wagons_today += ti['num_wagons']
            elif ti['return_day'] > day_t: 
                active_wagons_in_transit.append(ti)
        tracking_vars_sim['wagons_available'] += returned_wagons_today
        tracking_vars_sim['wagons_in_transit'] = active_wagons_in_transit

        if day_t == 1:
            curr_orig_load_caps = rem_load_caps_d1.copy()
            curr_dest_unload_caps = rem_unload_caps_d1.copy()
        else:
            curr_orig_load_caps = origins_df['daily_loading_capacity_tons'].copy()
            curr_dest_unload_caps = destinations_df['daily_unloading_capacity_tons'].copy()
        
        shipments_made_this_day = False
        
        if qmin_user_priority_order:
            qmin_daily_dest_iterator = [dest_id for dest_id in qmin_user_priority_order if dest_id in destinations_df.index]
        else:
            qmin_daily_dest_iterator = destinations_df.sort_values(by='q_min_initial_target_tons', ascending=False).index
        
        for dest_id in qmin_daily_dest_iterator:
            if dest_id not in destinations_df.index: continue
            q_min_initial_needed = destinations_df.loc[dest_id, 'q_min_initial_target_tons'] - destinations_df.loc[dest_id, 'q_min_initial_delivered_tons']
            if q_min_initial_needed <= EPSILON: continue
            
            relations_for_q_min = relations_df[relations_df['destination'] == dest_id].copy()
            if not relations_for_q_min.empty:
                relations_for_q_min = relations_for_q_min.merge(
                    origins_df[['current_available_product_tons']], left_on='origin', right_index=True
                ).sort_values(by='current_available_product_tons', ascending=False)

                for _, rel in relations_for_q_min.iterrows():
                    orig_id = rel['origin']; dist_km = rel['distance_km']
                    if q_min_initial_needed <= EPSILON: break
                    if curr_orig_load_caps.get(orig_id, 0) <= EPSILON or \
                       curr_dest_unload_caps.get(dest_id, 0) <= EPSILON or \
                       origins_df.loc[orig_id, 'current_available_product_tons'] <= EPSILON: continue
                    
                    shipped_qty, _, n_orig_cap, n_dest_cap = process_shipment(
                        day_t, orig_id, dest_id, dist_km, q_min_initial_needed,
                        origins_df, destinations_df, tracking_vars_sim,
                        curr_orig_load_caps[orig_id], curr_dest_unload_caps[dest_id], log_prefix="[QMIN_DAILY]"
                    )
                    if shipped_qty > EPSILON:
                        curr_orig_load_caps[orig_id] = n_orig_cap
                        curr_dest_unload_caps[dest_id] = n_dest_cap
                        destinations_df.loc[dest_id, 'q_min_initial_delivered_tons'] += shipped_qty
                        q_min_initial_needed -= shipped_qty
                        shipments_made_this_day = True
        
        if qmin_user_priority_order:
            phase2_dest_iterator = [
                dest_id for dest_id in qmin_user_priority_order 
                if dest_id in destinations_df.index and \
                   destinations_df.loc[dest_id, 'remaining_annual_demand_tons'] > EPSILON and \
                   (destinations_df.loc[dest_id, 'q_min_initial_delivered_tons'] >= (destinations_df.loc[dest_id, 'q_min_initial_target_tons'] - EPSILON))
            ]
        else: 
            phase2_dest_iterator = destinations_df[
                (destinations_df['remaining_annual_demand_tons'] > EPSILON) &
                (destinations_df['q_min_initial_delivered_tons'] >= (destinations_df['q_min_initial_target_tons'] - EPSILON))
            ].sort_values(by='remaining_annual_demand_tons', ascending=False).index

        for dest_id in phase2_dest_iterator:
            if dest_id not in destinations_df.index: continue
            if curr_dest_unload_caps.get(dest_id, 0) <= EPSILON: continue
            if destinations_df.loc[dest_id, 'remaining_annual_demand_tons'] <= EPSILON: continue
            
            best_origin_for_dest = None; best_origin_dist_km = 0; max_local_metric = -1.0 
            candidate_relations = profitable_relations_df[profitable_relations_df['destination'] == dest_id]
            
            for _, rel in candidate_relations.iterrows():
                orig_id = rel['origin']; dist_km = rel['distance_km']
                if orig_id not in origins_df.index: continue
                if origins_df.loc[orig_id, 'current_available_product_tons'] <= EPSILON or curr_orig_load_caps.get(orig_id, 0) <= EPSILON: continue
                
                qty_potential_stock = origins_df.loc[orig_id, 'current_available_product_tons']
                qty_potential_load_cap = curr_orig_load_caps.get(orig_id, 0)
                qty_potential_unload_cap = curr_dest_unload_caps.get(dest_id, 0)
                qty_potential_demand = destinations_df.loc[dest_id, 'remaining_annual_demand_tons']
                
                potential_qty_for_metric_calc = min(qty_potential_stock, qty_potential_load_cap, qty_potential_unload_cap, qty_potential_demand)
                if potential_qty_for_metric_calc < MIN_SHIPMENT_FOR_ONE_WAGON_TONS: continue
                
                current_local_metric = potential_qty_for_metric_calc 
                if current_local_metric > max_local_metric: 
                    max_local_metric = current_local_metric
                    best_origin_for_dest = orig_id; best_origin_dist_km = dist_km

            if best_origin_for_dest is not None:
                orig_to_ship_from = best_origin_for_dest; dist_for_shipment = best_origin_dist_km
                desired_std_qty = max_local_metric 
                
                shipped_qty, _, n_orig_cap, n_dest_cap = process_shipment(
                    day_t, orig_to_ship_from, dest_id, dist_for_shipment, desired_std_qty,
                    origins_df, destinations_df, tracking_vars_sim,
                    curr_orig_load_caps[orig_to_ship_from], curr_dest_unload_caps[dest_id], log_prefix="[SIM_PROFITABLE]"
                )
                if shipped_qty > EPSILON:
                    curr_orig_load_caps[orig_to_ship_from] = n_orig_cap
                    curr_dest_unload_caps[dest_id] = n_dest_cap
                    shipments_made_this_day = True
        
        all_total_dem_met = (destinations_df['remaining_annual_demand_tons'] <= EPSILON).all()
        if all_total_dem_met:
            if not silent_mode: print(f"\n--- Simulation terminée au jour {day_t}: Toutes les demandes annuelles sont satisfaites. ---")
            break
        
        if not shipments_made_this_day:
            no_prod_at_all = (origins_df['current_available_product_tons'] <= EPSILON).all()
            no_wagons_available_and_none_returning = (tracking_vars_sim['wagons_available'] == 0 and not tracking_vars_sim['wagons_in_transit'])
            
            if no_prod_at_all and no_wagons_available_and_none_returning:
                if not silent_mode: print(f"\n--- FIN Jour {day_t}: Plus de produit ET plus de wagons. ---")
                break
            elif no_prod_at_all:
                if (destinations_df['remaining_annual_demand_tons'] <= EPSILON).all() and \
                   (destinations_df['q_min_initial_delivered_tons'] >= (destinations_df['q_min_initial_target_tons'] - EPSILON)).all():
                     if not silent_mode: print(f"\n--- FIN Jour {day_t}: Plus de produit et toutes demandes satisfaites. ---")
                     break
                elif no_wagons_available_and_none_returning: 
                    if not silent_mode: print(f"\n--- FIN Jour {day_t}: Plus de produit ET plus de wagons (avec demandes restantes). ---")
                    break
            elif no_wagons_available_and_none_returning: 
                if not silent_mode: print(f"\n--- FIN Jour {day_t}: Plus de wagons (avec produit et demandes restantes). ---")
                break

    if day_t >= MAX_SIMULATION_DAYS and not all_total_dem_met:
        if not silent_mode: print(f"\n--- FIN: Limite de {MAX_SIMULATION_DAYS} jours atteinte. Demandes non toutes satisfaites. ---")

    shipments_summary_df = pd.DataFrame(tracking_vars_sim['shipments_log']) if tracking_vars_sim['shipments_log'] else pd.DataFrame()
    objective_value = calculate_objective_metric(shipments_summary_df, initial_relations_df_ref)
    
    return (shipments_summary_df, origins_df, destinations_df,
            initial_relations_df_ref, initial_destinations_df_ref, 
            tracking_vars_sim, objective_value)

# --- Fonctions pour le Recuit Simulé ---
def generate_random_neighbor(current_order):
    neighbor = list(current_order)
    n = len(neighbor)
    if n < 2: return neighbor
    i, j = random.sample(range(n), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor

def simulated_annealing_optimizer(initial_qmin_priority_order, initial_temp=100000.0, final_temp=0.1, alpha=0.99, max_iterations_at_temp=50):
    current_order = list(initial_qmin_priority_order)
    _, _, _, _, _, _, current_objective_value = run_simulation(qmin_user_priority_order=current_order, silent_mode=True)
    current_energy = -current_objective_value 
    
    print(f"Ordre QMIN initial pour Recuit Simulé: {current_order}")
    print(f"Métrique Tonnes*km initiale: {current_objective_value:.2f} (Énergie: {current_energy:.2f})")
    
    best_order = list(current_order)
    best_objective_value = current_objective_value
    best_energy = current_energy
    temp = initial_temp
    iteration_count = 0
    
    while temp > final_temp:
        for _i_iter_temp in range(max_iterations_at_temp): 
            iteration_count += 1
            neighbor_order = generate_random_neighbor(current_order)
            _, _, _, _, _, _, neighbor_objective_value = run_simulation(qmin_user_priority_order=neighbor_order, silent_mode=True)
            neighbor_energy = -neighbor_objective_value
            delta_energy = neighbor_energy - current_energy
            
            if delta_energy < 0: 
                current_order = neighbor_order; current_energy = neighbor_energy; current_objective_value = neighbor_objective_value
                if current_energy < best_energy: 
                    best_order = list(current_order); best_energy = current_energy; best_objective_value = current_objective_value
                    print(f"  *** Nouvelle meilleure solution globale trouvée à T={temp:.2f}, It_glob={iteration_count}: {best_order} -> Obj: {best_objective_value:.2f} ***")
            elif temp > EPSILON and random.random() < math.exp(-delta_energy / temp): 
                current_order = neighbor_order; current_energy = neighbor_energy; current_objective_value = neighbor_objective_value
            
        print(f"Fin palier T={temp:.2f}, It_glob={iteration_count}, Courant Obj: {current_objective_value:.2f} (Énergie: {current_energy:.2f}), Meilleur Obj: {best_objective_value:.2f}")
        temp *= alpha
            
    print(f"\nRecuit Simulé terminé après {iteration_count} itérations.")
    print(f"Meilleur ordre QMIN trouvé: {best_order}")
    return best_order, best_objective_value

# --- NOUVELLE FONCTION POUR L'EXPORT CSV ---
def export_results_to_csv(shipments_df, recap_dest_df, recap_orig_df):
    """Exporte les DataFrames de résultats dans des fichiers CSV."""
    print("\n--- Exportation des résultats vers des fichiers CSV ---")
    try:
        # Utiliser le point-virgule comme séparateur et la virgule pour les décimaux
        # pour une meilleure compatibilité avec Excel en français.
        if not shipments_df.empty:
            shipments_df.to_csv('log_expeditions.csv', index=False, sep=';', decimal=',')
            print(" -> Fichier 'log_expeditions.csv' créé avec succès.")

        if not recap_dest_df.empty:
            recap_dest_df.to_csv('recapitulatif_destinations.csv', index=True, sep=';', decimal=',')
            print(" -> Fichier 'recapitulatif_destinations.csv' créé avec succès.")

        if not recap_orig_df.empty:
            recap_orig_df.to_csv('recapitulatif_origines.csv', index=True, sep=';', decimal=',')
            print(" -> Fichier 'recapitulatif_origines.csv' créé avec succès.")

    except Exception as e:
        print(f"ERREUR lors de l'exportation en CSV : {e}")

# --- Fonctions utilitaires pour le menu ---
def get_qmin_priority_options(temp_dest_df_for_options, available_dest_ids):
    if 'q_min_initial_target_tons' not in temp_dest_df_for_options.columns:
        temp_dest_df_for_options['q_min_initial_target_tons'] = 0.20 * temp_dest_df_for_options.get('annual_demand_tons', 0)
    if 'annual_demand_tons' not in temp_dest_df_for_options.columns:
        temp_dest_df_for_options['annual_demand_tons'] = 0
    if 'min_distance_km' not in temp_dest_df_for_options.columns:
        temp_dest_df_for_options['min_distance_km'] = float('inf')
        
    options_map = {
        '1': temp_dest_df_for_options.sort_values(by='q_min_initial_target_tons', ascending=False).index.tolist(),
        '2': temp_dest_df_for_options.sort_values(by='q_min_initial_target_tons', ascending=True).index.tolist(),
        '3': temp_dest_df_for_options.sort_values(by='annual_demand_tons', ascending=False).index.tolist(),
        '4': temp_dest_df_for_options.sort_values(by='annual_demand_tons', ascending=True).index.tolist(),
        '5': temp_dest_df_for_options.sort_values(by='min_distance_km', ascending=True).index.tolist(),
        '6': temp_dest_df_for_options.sort_values(by='min_distance_km', ascending=False).index.tolist(),
        '8': available_dest_ids 
    }
    return options_map

def get_user_choice_for_qmin_order(temp_dest_df_for_options, available_dest_ids, prompt_message=""):
    if prompt_message: print(prompt_message)
    print("1. Par 'Objectif QMIN (20%)' décroissant")
    print("2. Par 'Objectif QMIN (20%)' croissant")
    print("3. Par 'Demande Annuelle Totale' décroissante")
    print("4. Par 'Demande Annuelle Totale' croissante")
    print("5. Par 'Distance minimale vers destination' croissante")
    print("6. Par 'Distance minimale vers destination' décroissante")
    print("7. Ordre spécifique que vous allez entrer")
    print("8. Aucun ordre spécifique (ordre des données initiales)")
    
    options_map = get_qmin_priority_options(temp_dest_df_for_options, available_dest_ids)
    while True:
        choice_str = input("Votre choix (1-8) : ").strip()
        if choice_str in options_map: 
            return [str(item).strip() for item in options_map[choice_str]]
        elif choice_str == '7':
            while True:
                user_input_order_str = input(f"Entrez l'ordre des destinations séparées par des virgules (ex: y2,y1,y3). Valides: {', '.join(available_dest_ids)} : ").strip()
                if not user_input_order_str: continue
                potential_order = [dest.strip() for dest in user_input_order_str.split(',')]
                invalid_dests = [d for d in potential_order if d not in available_dest_ids]
                if len(potential_order) != len(set(potential_order)):
                    print("Erreur: L'ordre entré contient des doublons.")
                    continue
                if not invalid_dests and len(potential_order) == len(available_dest_ids): 
                    return potential_order
                elif invalid_dests:
                    print(f"Erreur: Destinations non valides : {', '.join(invalid_dests)}.")
                else: 
                    print(f"Erreur: Veuillez spécifier un ordre pour toutes les destinations ({len(available_dest_ids)} attendues, {len(potential_order)} fournies).")
        else: print("Choix invalide.")

# --- Exécution et Affichage ---
if __name__ == '__main__':
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', 500)
    
    ref_relations_df, ref_origins_df, ref_destinations_df = load_data()
    
    temp_dest_df_for_options = ref_destinations_df.copy()
    temp_dest_df_for_options['q_min_initial_target_tons'] = 0.20 * temp_dest_df_for_options['annual_demand_tons']
    min_distances = ref_relations_df.groupby('destination')['distance_km'].min().rename('min_distance_km')
    temp_dest_df_for_options = temp_dest_df_for_options.merge(min_distances, left_index=True, right_index=True, how='left')
    temp_dest_df_for_options['min_distance_km'].fillna(float('inf'), inplace=True)
    available_dest_ids = list(temp_dest_df_for_options.index) 
    
    final_qmin_order_to_run = None
    objective_value_for_final_run = None 
    
    print(f"Destinations disponibles pour la priorité QMIN: {', '.join(available_dest_ids)}")
    print("\nStratégie pour livraison QMIN (20%):")
    main_menu_options_text = "\n".join([
        "1-6, 8: Tris prédéfinis / Ordre initial (Exécute la simulation directement)",
        "7: Ordre spécifique manuel (Exécute la simulation directement)",
        "9: Optimiser l'ordre QMIN par Recuit Simulé (Maximiser Tonnes*km)"
    ])
    print(main_menu_options_text)
    
    user_choice_valid = False
    while not user_choice_valid:
        main_choice_str = input("Votre choix principal (1-9) : ").strip()
        qmin_options_map = get_qmin_priority_options(temp_dest_df_for_options, available_dest_ids)
        
        if main_choice_str in qmin_options_map: 
            final_qmin_order_to_run = qmin_options_map[main_choice_str]
            user_choice_valid = True
        elif main_choice_str == '7':
            final_qmin_order_to_run = get_user_choice_for_qmin_order(temp_dest_df_for_options, available_dest_ids, "")
            user_choice_valid = True
        elif main_choice_str == '9':
            print("\n--- Lancement de l'Optimisation de l'Ordre QMIN par Recuit Simulé ---")
            start_order_for_optimizer = get_user_choice_for_qmin_order(temp_dest_df_for_options, available_dest_ids, 
                "Veuillez choisir un ordre de priorité QMIN DE DÉPART pour le Recuit Simulé :")
            print(f"\nRecuit Simulé à partir de l'ordre QMIN : {start_order_for_optimizer}")
            
            initial_temp = 10000.0  
            final_temp = 10.0       
            alpha = 0.5
            iterations_per_temp = 10
            
            print(f"Paramètres Recuit Simulé: T_init={initial_temp}, T_final={final_temp}, alpha={alpha}, iter_par_temp={iterations_per_temp}")
            final_qmin_order_to_run, objective_value_for_final_run = simulated_annealing_optimizer(
                start_order_for_optimizer, initial_temp=initial_temp, final_temp=final_temp, 
                alpha=alpha, max_iterations_at_temp=iterations_per_temp)
            user_choice_valid = True
        else: print("Choix invalide.")
        
    print(f"\nOrdre QMIN final pour simulation détaillée : {final_qmin_order_to_run}")
    if objective_value_for_final_run is not None: 
        print(f"Métrique Tonnes*km pour cet ordre (obtenu par Recuit Simulé) : {objective_value_for_final_run:.2f}")
        
    print(f"\nExécution de la simulation finale avec l'ordre QMIN : {final_qmin_order_to_run}")
    print(f"L'ordre pour les expéditions standard (Phase 2) sera DÉRIVÉ DE CET ORDRE QMIN.")
    
    (shipments_df, final_origins_df, final_destinations_df,
     _returned_initial_relations_df, _returned_initial_destinations_df, 
     final_tracking_vars, final_objective_metric_value) = run_simulation(
        qmin_user_priority_order=final_qmin_order_to_run, silent_mode=False)
    
    if not shipments_df.empty:
        qmin_shipments_df = shipments_df[shipments_df['type'].isin(['[QMIN_INIT_J1]', '[QMIN_DAILY]'])].copy()
        if not qmin_shipments_df.empty:
            qmin_daily_total_to_yj = qmin_shipments_df.groupby(['ship_day', 'destination'])['quantity_tons'].sum().rename('Q_Qmin_Recu_Yj_Jr')
            qmin_daily_total_from_xi = qmin_shipments_df.groupby(['ship_day', 'origin'])['quantity_tons'].sum().rename('Q_Qmin_Envoye_Xi_Jr')
            
            qmin_shipments_df = qmin_shipments_df.merge(ref_destinations_df[['annual_demand_tons']], left_on='destination', right_index=True, how='left').rename(columns={'annual_demand_tons': 'Dem_Ann_Yj_Initial'})
            qmin_shipments_df = qmin_shipments_df.merge(ref_relations_df[['origin', 'destination', 'profitability']], on=['origin', 'destination'], how='left').rename(columns={'profitability': 'Rentab_Relation_Utilisee'})
            qmin_shipments_df['Rentab_Relation_Utilisee'] = qmin_shipments_df['Rentab_Relation_Utilisee'].map({1: 'Oui', 0: 'Non', pd.NA: 'N/A'})
            
            qmin_shipments_df = qmin_shipments_df.merge(qmin_daily_total_to_yj, on=['ship_day', 'destination'], how='left')
            qmin_shipments_df = qmin_shipments_df.merge(qmin_daily_total_from_xi, on=['ship_day', 'origin'], how='left')
            
            print("\n--- Matrice des Expéditions QMIN (Vue Origine) ---")
            matrice1_qmin = qmin_shipments_df[['origin', 'ship_day', 'destination', 'quantity_tons', 'Dem_Ann_Yj_Initial', 'Rentab_Relation_Utilisee', 'Q_Qmin_Recu_Yj_Jr', 'type']].copy()
            matrice1_qmin.rename(columns={'origin': 'Orig', 'ship_day': 'Jr_Exp', 'destination': 'Dest', 
                                          'quantity_tons': 'Q_Exp_QMIN_Xi_Dest', 'Q_Qmin_Recu_Yj_Jr': 'QminRecuJr_Yj_Total'}, inplace=True)
            print(matrice1_qmin.sort_values(by=['Jr_Exp', 'Orig', 'Dest']))
            
            print("\n--- Matrice des Expéditions QMIN (Vue Destination) ---")
            matrice2_qmin = qmin_shipments_df[['destination', 'ship_day', 'origin', 'quantity_tons', 'Dem_Ann_Yj_Initial', 'Rentab_Relation_Utilisee', 'Q_Qmin_Envoye_Xi_Jr', 'type']].copy()
            matrice2_qmin.rename(columns={'destination': 'Dest', 'ship_day': 'Jr_Exp', 'origin': 'Orig', 
                                          'quantity_tons': 'Q_Exp_QMIN_Dest_Orig', 'Q_Qmin_Envoye_Xi_Jr': 'QminEnvoyeJr_Xi_Total'}, inplace=True)
            print(matrice2_qmin.sort_values(by=['Jr_Exp', 'Dest', 'Orig']))
        else: print("Aucune expédition de type QMIN (Initiale ou Quotidienne) n'a été effectuée.")
        
        print("\n--- Toutes les Expéditions (Détail) ---")
        print(shipments_df.sort_values(by=['ship_day', 'origin', 'destination']))
    else: print("Aucune expédition (QMIN ou autre) à afficher en détail.")

    print("\n--- Récapitulatif Final par Destination (Yj) ---")
    recap_dest_df = final_destinations_df.copy()
    recap_dest_df.index.name = 'Dest_Yj'
    recap_dest_df = recap_dest_df.merge(ref_destinations_df[['annual_demand_tons']].rename(columns={'annual_demand_tons':'Dem_Ann_Yj_Orig'}),
                                        left_index=True, right_index=True, how='left')
    recap_dest_df = recap_dest_df.rename(columns={'q_min_initial_target_tons': 'Qmin_Cible_Yj', 
                                                 'q_min_initial_delivered_tons': 'Qmin_Livre_Yj', 
                                                 'delivered_so_far_tons': 'Total_Livre_Yj', 
                                                 'remaining_annual_demand_tons': 'Dem_Rest_Yj'})
    print(recap_dest_df[['Dem_Ann_Yj_Orig', 'Qmin_Cible_Yj', 'Qmin_Livre_Yj', 'Total_Livre_Yj', 'Dem_Rest_Yj']].round(2))

    print("\n--- Récapitulatif Final par Origine (Xi) ---")
    recap_orig_df = final_origins_df.copy()
    if not shipments_df.empty:
        total_expedie_par_xi = shipments_df.groupby('origin')['quantity_tons'].sum().rename('Total_Exp_Xi')
        recap_orig_df = recap_orig_df.merge(total_expedie_par_xi, left_index=True, right_index=True, how='left')
        recap_orig_df['Total_Exp_Xi'] = recap_orig_df['Total_Exp_Xi'].fillna(0)
    else: 
        recap_orig_df['Total_Exp_Xi'] = 0.0
    recap_orig_df.index.name = 'Orig_Xi'
    recap_orig_df = recap_orig_df.merge(ref_origins_df[['initial_available_product_tons']].rename(columns={'initial_available_product_tons':'Stock_Init_Xi_Orig'}),
                                        left_index=True, right_index=True, how='left')
    recap_orig_df = recap_orig_df.rename(columns={'current_available_product_tons': 'Stock_Fin_Xi'})
    print(recap_orig_df[['Stock_Init_Xi_Orig', 'Total_Exp_Xi', 'Stock_Fin_Xi']].round(2))
    
    print(f"\n--- Informations Finales sur les Wagons ---")
    initial_wagons_value = final_tracking_vars.get('initial_wagons', 0)
    print(f"Wagons initiaux: {initial_wagons_value}")
    print(f"Wagons restants disponibles à la fin: {final_tracking_vars.get('wagons_available', 'N/A')}")
    num_wagons_still_in_transit = sum(w['num_wagons'] for w in final_tracking_vars.get('wagons_in_transit', []))
    print(f"Wagons encore en transit à la fin: {num_wagons_still_in_transit}")
    
    print(f"\nIndicateur global final (Tonnes * km) pour la simulation affichée : {final_objective_metric_value:.2f}")
    if not shipments_df.empty:
        temp_metric_df = shipments_df.groupby(['origin', 'destination'])['quantity_tons'].sum().reset_index()
        temp_metric_df.rename(columns={'quantity_tons': 'total_quantity_on_relation'}, inplace=True)
        temp_metric_df = pd.merge(temp_metric_df, ref_relations_df[['origin', 'destination', 'distance_km']], on=['origin', 'destination'], how='left')
        temp_metric_df.fillna({'distance_km': 0}, inplace=True) 
        temp_metric_df['tonnes_km_relation'] = temp_metric_df['total_quantity_on_relation'] * temp_metric_df['distance_km']
        print("\n--- Détail de l'Indicateur Tonnes*km par Relation (Simulation Finale Détaillée) ---")
        print(temp_metric_df[['origin', 'destination', 'total_quantity_on_relation', 'distance_km', 'tonnes_km_relation']].round(2).sort_values(by=['origin','destination']))
    else: print("Aucune expédition pour le détail de la métrique Tonnes*km.")

    # --- PARTIE AJOUTÉE POUR L'EXPORTATION ---
    while True:
        export_choice = input("\nVoulez-vous exporter ces récapitulatifs en fichiers CSV ? (o/n) : ").strip().lower()
        if export_choice in ['o', 'oui']:
            export_results_to_csv(shipments_df, recap_dest_df, recap_orig_df)
            break
        elif export_choice in ['n', 'non']:
            print("Exportation annulée.")
            break
        else:
            print("Choix invalide. Veuillez entrer 'o' ou 'n'.")
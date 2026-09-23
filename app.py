import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from designer_core import (
    ScenarioDefinition,
    JobDefinition,
    OperationStep,
    MachineGroup,
    get_preset_3x3,
    get_preset_7x6,
    generate_random_scenario,
    compute_workload_distribution,
    validate_scenario,
    export_to_python_code,
    export_to_json,
    to_jobs_data,
    rename_machine_group,
    generate_factory_graph_dot,
    generate_neural_network_dot
)

# Set page configuration
st.set_page_config(
    page_title="JobRL — Visual Scenario Designer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .metric-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "scenario" not in st.session_state:
    st.session_state.scenario = get_preset_7x6()

scenario: ScenarioDefinition = st.session_state.scenario


# ==============================================================================
# Top App Header
# ==============================================================================
st.markdown('<div class="main-header">🏭 JobRL — Visual Scenario Designer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Design custom factory floor setups, work centers, and digital twin parameters '
    'without coding. Export ready-to-use parameter files for your simulation and numerical methods scripts.</div>',
    unsafe_allow_html=True
)


# ==============================================================================
# Sidebar: Quick Presets & Export Actions
# ==============================================================================
with st.sidebar:
    st.header("⚡ Scenario Presets")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("Load 3×3 Toy", use_container_width=True):
            st.session_state.scenario = get_preset_3x3()
            st.rerun()
    with col_p2:
        if st.button("Load 7×6 Bench", use_container_width=True):
            st.session_state.scenario = get_preset_7x6()
            st.rerun()

    st.markdown("---")
    st.subheader("🎲 Random Generator")
    with st.expander("Configure & Generate", expanded=False):
        gen_jobs = st.slider("Number of Jobs", 2, 15, 5)
        gen_machines = st.slider("Number of Machines", 2, 12, 4)
        gen_mode = st.selectbox("Mode", ["simplified", "flexible"])
        gen_groups = st.slider("Machine Groups", 1, min(gen_machines, 5), 2) if gen_mode == "flexible" else 1
        dur_range = st.slider("Duration Range (s)", 1.0, 15.0, (1.0, 6.0))
        
        if st.button("Generate Scenario", type="primary", use_container_width=True):
            st.session_state.scenario = generate_random_scenario(
                num_jobs=gen_jobs,
                num_machines=gen_machines,
                mode=gen_mode,
                min_dur=dur_range[0],
                max_dur=dur_range[1],
                num_groups=gen_groups
            )
            st.success(f"Generated {gen_jobs}×{gen_machines} ({gen_mode}) scenario!")
            st.rerun()

    st.markdown("---")
    st.subheader("💾 Quick Export")
    
    is_valid, errors, warnings = validate_scenario(scenario)
    if is_valid:
        st.success("✅ Scenario is Valid & Ready")
    else:
        st.error(f"⚠️ {len(errors)} validation issue(s) detected")

    py_code = export_to_python_code(scenario)
    json_data = export_to_json(scenario)

    st.download_button(
        label="📥 Download scenario_parameters.py",
        data=py_code,
        file_name="scenario_parameters.py",
        mime="text/x-python",
        use_container_width=True,
        type="primary" if is_valid else "secondary"
    )

    st.download_button(
        label="📥 Download scenario.json",
        data=json_data,
        file_name="scenario.json",
        mime="application/json",
        use_container_width=True
    )


# ==============================================================================
# Main Content: Navigation Tabs
# ==============================================================================
tab_factory, tab_jobs, tab_physics, tab_network, tab_inspect, tab_export = st.tabs([
    "1. 🏭 Factory & Work Centers",
    "2. 📋 Job Routing Builder",
    "3. ⚙️ Digital Twin Physics",
    "4. 🕸️ Machine & Network Topology",
    "5. 📊 Workload & Validation",
    "6. 💾 Export & Code Snippets"
])


# ==============================================================================
# Tab 1: Factory & Work Centers (Machine Groups)
# ==============================================================================
with tab_factory:
    st.subheader("Factory Floor & Work Center Configuration")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        new_num_machines = st.number_input(
            "Total Number of Machines",
            min_value=1,
            max_value=20,
            value=scenario.num_machines,
            step=1,
            help="Total number of physical machines available on the shop floor."
        )
        if new_num_machines != scenario.num_machines:
            scenario.num_machines = new_num_machines
            st.rerun()

    with col2:
        new_mode = st.radio(
            "Scheduling Paradigm / Mode",
            ["simplified", "flexible"],
            index=0 if scenario.mode == "simplified" else 1,
            format_func=lambda x: "Simplified (Classic JSS: 1 machine per step)" if x == "simplified" else "Flexible (FJSP: alternative machines / work centers)",
            help="In Flexible mode, operations can be assigned to Work Centers with multiple alternative machines."
        )
        if new_mode != scenario.mode:
            scenario.mode = new_mode
            scenario.factory_config["mode"] = new_mode
            st.rerun()

    st.markdown("---")
    st.subheader("🛠️ Work Centers / Machine Groups")
    st.caption("Group physical machines into logical work centers (e.g., Lathes, CNC Mills, Inspection Stations). In Flexible mode, jobs can be dispatched to any machine in a group.")

    # Display existing machine groups
    if scenario.machine_groups:
        cols = st.columns(min(3, max(1, len(scenario.machine_groups))))
        for idx, (g_name, group) in enumerate(list(scenario.machine_groups.items())):
            col = cols[idx % len(cols)]
            with col:
                with st.container(border=True):
                    st.markdown(f"""
                    <div style="border-left: 4px solid {group.color}; padding-left: 0.5rem; margin-bottom: 0.5rem;">
                        <strong style="font-size: 1.05rem;">{group.name}</strong><br>
                        <span style="font-size: 0.85rem; color: #4B5563;">Assigned: {[f'M{m}' for m in group.machine_ids]}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("✏️ Rename / Edit", expanded=False):
                        edited_name = st.text_input("Name", value=group.name, key=f"edit_name_{g_name}")
                        available_m = list(range(scenario.num_machines))
                        edited_machines = st.multiselect(
                            "Machines",
                            available_m,
                            default=[m for m in group.machine_ids if m in available_m],
                            format_func=lambda m: f"Machine {m}",
                            key=f"edit_mach_{g_name}"
                        )
                        edited_color = st.color_picker("Color", value=group.color, key=f"edit_col_{g_name}")
                        
                        col_save, col_del = st.columns(2)
                        with col_save:
                            if st.button("Save", key=f"save_g_{g_name}", type="primary"):
                                if edited_name != g_name:
                                    success = rename_machine_group(scenario, g_name, edited_name)
                                    if not success:
                                        st.error("Name already exists or is empty.")
                                        st.stop()
                                    target_group = scenario.machine_groups[edited_name]
                                else:
                                    target_group = group
                                target_group.machine_ids = edited_machines
                                target_group.color = edited_color
                                st.success(f"Updated '{edited_name}'!")
                                st.rerun()
                        with col_del:
                            if st.button("Delete", key=f"del_group_{g_name}"):
                                del scenario.machine_groups[g_name]
                                st.rerun()
    else:
        st.info("No work centers defined yet. Add one below!")

    with st.expander("➕ Add New Work Center", expanded=False):
        with st.form("add_group_form"):
            new_g_name = st.text_input("Work Center Name", placeholder="e.g., Primary Milling")
            available_machines = list(range(scenario.num_machines))
            assigned_m = st.multiselect("Assigned Machines", available_machines, default=[0] if available_machines else [])
            color_choice = st.color_picker("Color Badge", "#3B82F6")
            submit_g = st.form_submit_button("Add Work Center")
            
            if submit_g:
                if not new_g_name:
                    st.error("Please enter a name for the work center.")
                elif not assigned_m:
                    st.error("Please assign at least one machine to the work center.")
                else:
                    scenario.machine_groups[new_g_name] = MachineGroup(
                        name=new_g_name,
                        machine_ids=assigned_m,
                        color=color_choice
                    )
                    st.success(f"Work Center '{new_g_name}' added!")
                    st.rerun()


# ==============================================================================
# Tab 2: Job Routing Builder
# ==============================================================================
with tab_jobs:
    st.subheader("Job & Operation Routing Builder")
    st.caption("Configure the sequential steps for each job. Add operations, select the required machine or work center, and set processing durations.")

    col_top1, col_top2 = st.columns([1, 1])
    with col_top1:
        st.markdown(f"**Total Jobs Defined:** `{len(scenario.jobs)}`")
    with col_top2:
        if st.button("➕ Add New Job", type="primary"):
            new_j_id = len(scenario.jobs)
            new_job = JobDefinition(
                job_id=new_j_id,
                name=f"Job {new_j_id}",
                steps=[OperationStep(step_idx=0, machine_id=0, duration=3.0)]
            )
            scenario.jobs.append(new_job)
            scenario.num_jobs = len(scenario.jobs)
            st.rerun()

    st.markdown("---")

    # List Jobs
    for j_idx, job in enumerate(scenario.jobs):
        with st.expander(f"📦 {job.name} ({len(job.steps)} Operations)", expanded=(j_idx == 0)):
            col_j1, col_j2, col_j3 = st.columns([2, 2, 1])
            with col_j1:
                job.name = st.text_input("Job Name", value=job.name, key=f"jname_{j_idx}")
            with col_j2:
                job.release_time = st.number_input("Release Time (s)", min_value=0.0, value=float(job.release_time), step=0.5, key=f"jrel_{j_idx}")
            with col_j3:
                st.write("")
                st.write("")
                if st.button("🗑️ Delete Job", key=f"del_job_{j_idx}"):
                    scenario.jobs.pop(j_idx)
                    # Re-index
                    for idx, j in enumerate(scenario.jobs):
                        j.job_id = idx
                    scenario.num_jobs = len(scenario.jobs)
                    st.rerun()

            st.markdown("**Sequential Operations:**")
            
            # Step editor table
            for s_idx, step in enumerate(job.steps):
                c_step_num, c_step_target, c_step_dur, c_step_act = st.columns([1, 3, 2, 1])
                
                with c_step_num:
                    st.markdown(f"**Step {s_idx + 1}**")
                
                with c_step_target:
                    if scenario.mode == "simplified":
                        # Machine dropdown
                        machine_choices = list(range(scenario.num_machines))
                        current_m = step.machine_id if step.machine_id in machine_choices else 0
                        chosen_m = st.selectbox(
                            "Machine",
                            machine_choices,
                            index=machine_choices.index(current_m),
                            format_func=lambda m: f"Machine {m}",
                            key=f"step_m_{j_idx}_{s_idx}",
                            label_visibility="collapsed"
                        )
                        step.machine_id = chosen_m
                    else:
                        # Work Center or Machine Choice
                        group_options = list(scenario.machine_groups.keys())
                        if group_options:
                            curr_g = step.group_name if step.group_name in group_options else group_options[0]
                            chosen_g = st.selectbox(
                                "Work Center",
                                group_options,
                                index=group_options.index(curr_g),
                                key=f"step_g_{j_idx}_{s_idx}",
                                label_visibility="collapsed"
                            )
                            step.group_name = chosen_g
                        else:
                            st.warning("Define a Work Center in Tab 1 first.")

                with c_step_dur:
                    chosen_dur = st.number_input(
                        "Duration (s)",
                        min_value=0.1,
                        max_value=100.0,
                        value=float(step.duration),
                        step=0.5,
                        key=f"step_dur_{j_idx}_{s_idx}",
                        label_visibility="collapsed"
                    )
                    step.duration = chosen_dur

                with c_step_act:
                    if st.button("✕", key=f"del_step_{j_idx}_{s_idx}", help="Remove this operation"):
                        job.steps.pop(s_idx)
                        for idx, s in enumerate(job.steps):
                            s.step_idx = idx
                        st.rerun()

            if st.button(f"➕ Add Step to {job.name}", key=f"add_step_{j_idx}"):
                new_s_idx = len(job.steps)
                default_g = list(scenario.machine_groups.keys())[0] if scenario.machine_groups else None
                job.steps.append(OperationStep(
                    step_idx=new_s_idx,
                    machine_id=0,
                    group_name=default_g,
                    duration=3.0
                ))
                st.rerun()


# ==============================================================================
# Tab 3: Digital Twin Physics
# ==============================================================================
with tab_physics:
    st.subheader("Digital Twin Physics & Floor Settings")
    st.caption("Configure stochastic machine failures, dynamic job arrivals, buffer limits, and setup times. These will be exported into FACTORY_CONFIG.")

    cfg = scenario.factory_config

    col_ph1, col_ph2 = st.columns(2)

    with col_ph1:
        st.markdown("#### ⚡ Machine Breakdowns")
        cfg["enable_breakdowns"] = st.toggle(
            "Enable Stochastic Breakdowns",
            value=bool(cfg.get("enable_breakdowns", False)),
            help="Simulates machine failures governed by exponential MTTF and MTTR distributions."
        )
        if cfg["enable_breakdowns"]:
            cfg["failure_rate_lambda"] = st.number_input(
                "Failure Rate (λ failure)",
                min_value=0.0001,
                max_value=0.1,
                value=float(cfg.get("failure_rate_lambda", 0.005)),
                format="%.4f",
                help="Exponential rate for MTTF (Mean Time To Failure = 1/λ)."
            )
            cfg["repair_mean_mu"] = st.number_input(
                "Mean Repair Duration (μ repair)",
                min_value=1.0,
                max_value=100.0,
                value=float(cfg.get("repair_mean_mu", 10.0)),
                step=1.0,
                help="Expected duration (seconds) required to service and repair a broken machine."
            )

        st.markdown("#### 📦 Machine Buffer Limits")
        cfg["enable_buffer_limits"] = st.toggle(
            "Enable Finite Buffers",
            value=bool(cfg.get("enable_buffer_limits", False)),
            help="Enforces a maximum waiting queue capacity in front of each machine."
        )
        if cfg["enable_buffer_limits"]:
            cfg["buffer_capacity"] = st.number_input(
                "Buffer Queue Capacity",
                min_value=1,
                max_value=50,
                value=int(cfg.get("buffer_capacity", 5)),
                step=1
            )

    with col_ph2:
        st.markdown("#### ⏱️ Dynamic Job Arrivals")
        cfg["enable_dynamic_arrivals"] = st.toggle(
            "Enable Dynamic Poisson Arrivals",
            value=bool(cfg.get("enable_dynamic_arrivals", False)),
            help="Jobs arrive dynamically throughout the day rather than all being available at t=0."
        )
        if cfg["enable_dynamic_arrivals"]:
            cfg["arrival_rate_lambda"] = st.number_input(
                "Arrival Rate (λ arrival)",
                min_value=0.001,
                max_value=0.5,
                value=float(cfg.get("arrival_rate_lambda", 0.02)),
                format="%.4f",
                help="Poisson process parameter for job inter-arrival times."
            )

        st.markdown("#### 🔧 Sequence-Dependent Setup Times (SDST)")
        cfg["enable_setup_times"] = st.toggle(
            "Enable Setup Times",
            value=bool(cfg.get("enable_setup_times", False)),
            help="Applies a tooling setup delay whenever a machine switches between different job types."
        )

        st.markdown("#### 🎯 RL Reward Objective")
        cfg["reward_type"] = st.selectbox(
            "Reward Formulation",
            ["dense_idle_delay", "sparse_makespan"],
            index=0 if cfg.get("reward_type") == "dense_idle_delay" else 1,
            help="'dense_idle_delay' penalizes elapsed time and machine idle starvation. 'sparse_makespan' rewards solely at completion."
        )


# ==============================================================================
# Tab 4: Machine & Network Topology (Interactive Plant Visualizer)
# ==============================================================================
with tab_network:
    st.subheader("🕸️ Factory Floor Network & Node Topology")
    st.caption(
        "Explore your factory setup represented as a **Neural Network structure**: "
        "each horizontal row represents a manufacturing Work Center layer, while individual neurons within that layer represent physical machines. "
        "Synaptic edges trace jobs flowing forward through the factory, with an optional Maintenance & Diagnostics Bay showing stochastic failure loops."
    )

    # Visualization Architecture Selector
    col_v1, col_v2 = st.columns([2, 2])
    with col_v1:
        arch_choice = st.selectbox(
            "Visualization Architecture",
            [
                "🧠 Neural Network Layered View (Rows = Work Centers, Neurons = Machines)",
                "🏭 Clustered Work Center View (Bounding Box Enclosures)",
                "🌐 Macro Flow View (High-Level Work Center Transitions)"
            ],
            index=0,
            help="Choose between a Neural Network layered topology or traditional plant cluster layout."
        )
    with col_v2:
        route_filter_opts = ["All Jobs Combined"] + [f"Highlight {j.name}" for j in scenario.jobs]
        selected_route_filter = st.selectbox(
            "Process Route Filter",
            route_filter_opts,
            index=0,
            key="network_route_filter",
            help="Choose 'All Jobs Combined' to see overall machine traffic, or isolate a single job's route."
        )
        highlight_id = None
        if selected_route_filter != "All Jobs Combined":
            highlight_name = selected_route_filter.replace("Highlight ", "")
            for j in scenario.jobs:
                if j.name == highlight_name:
                    highlight_id = j.job_id
                    break

    # Neural Network Layout & Density Options
    if "Neural Network" in arch_choice:
        col_ori1, col_ori2 = st.columns([2, 2])
        with col_ori1:
            orient_choice = st.radio(
                "Layer Orientation",
                [
                    "⬇️ Horizontal Rows (Top-to-Bottom, Neural Net Layers)",
                    "➡️ Vertical Columns (Left-to-Right)"
                ],
                index=0,
                horizontal=True,
                help="In Horizontal Rows mode, each layer forms a horizontal row of machines with vertical synaptic flow downwards."
            )
        with col_ori2:
            dense_conn = st.checkbox(
                "Full Synaptic Mesh (Dense Bipartite Connections)",
                value=True,
                help="In flexible mode, renders the full bipartite synaptic web between consecutive layers."
            )
    else:
        orient_choice = "⬇️ Horizontal Rows (Top-to-Bottom, Neural Net Layers)"
        dense_conn = True

    # Badge & Detail Toggles
    with st.expander("🛠️ Display Badges & Stochastic Filters", expanded=False):
        c_tog1, c_tog2, c_tog3, c_tog4 = st.columns(4)
        with c_tog1:
            show_bd = st.checkbox("Show Breakdowns (⚡ MTTF/MTTR)", value=True, key="net_show_bd")
        with c_tog2:
            show_hub = st.checkbox("Show Repair Bay Node (🔧)", value=True, key="net_show_hub")
        with c_tog3:
            show_buf = st.checkbox("Show Buffer Limits (📦)", value=True, key="net_show_buf")
        with c_tog4:
            show_wl = st.checkbox("Show Workload (s)", value=True, key="net_show_wl")

    # Render according to chosen architecture
    if "Neural Network" in arch_choice:
        chosen_ori = "TB" if "Horizontal" in orient_choice else "LR"
        dot_source = generate_neural_network_dot(
            scenario,
            highlight_job_id=highlight_id,
            orientation=chosen_ori,
            show_breakdown_info=show_bd,
            show_buffer_info=show_buf,
            show_workload_info=show_wl,
            show_repair_hub=show_hub,
            dense_connections=dense_conn
        )
    elif "Clustered" in arch_choice:
        dot_source = generate_factory_graph_dot(
            scenario,
            highlight_job_id=highlight_id,
            layout_direction="LR",
            show_breakdown_info=show_bd,
            show_buffer_info=show_buf,
            show_workload_info=show_wl,
            view_mode="detailed"
        )
    else:
        dot_source = generate_factory_graph_dot(
            scenario,
            highlight_job_id=highlight_id,
            layout_direction="LR",
            show_breakdown_info=show_bd,
            show_buffer_info=show_buf,
            show_workload_info=show_wl,
            view_mode="macro_work_centers"
        )

    st.graphviz_chart(dot_source, use_container_width=True)

    # Neural Network Visual Legend
    if "Neural Network" in arch_choice:
        st.markdown("""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.5rem 0.8rem; margin-top: 0.3rem; font-size: 0.82rem; color: #475569;">
            <strong>📘 Neural Network Visual Legend:</strong> &nbsp;
            <span>⚪ <strong>Circular Neurons</strong> = Physical Machines</span> &nbsp;|&nbsp; 
            <span style="color: #6366F1;">━ <strong>Forward Synapses</strong></span> &nbsp;|&nbsp; 
            <span style="color: #8B5CF6;">╍ <strong>Skip Shortcuts</strong> (ResNet)</span> &nbsp;|&nbsp; 
            <span style="color: #F43F5E;">┄ <strong>Recurrent Loops</strong> (RNN)</span> &nbsp;|&nbsp; 
            <span style="color: #EC4899;">━ <strong>Active Job Path</strong></span> &nbsp;|&nbsp; 
            <span style="color: #EF4444;">🔴 <strong>Bottleneck Neuron</strong></span>
        </div>
        """, unsafe_allow_html=True)

    # Topology Summary Cards
    st.markdown("---")
    st.markdown("#### 🏭 Layer & Work Center Architecture Summary")
    if scenario.machine_groups:
        cols = st.columns(min(4, max(1, len(scenario.machine_groups))))
        group_items_list = list(scenario.machine_groups.items())
        total_layers = len(group_items_list)
        for idx, (g_name, group) in enumerate(group_items_list):
            if idx == 0:
                stage_badge = "🟢 Input Layer (Stage 1)"
            elif idx == total_layers - 1 and total_layers > 1:
                stage_badge = f"🟠 Output Layer (Stage {idx+1})"
            else:
                stage_badge = f"🔵 Hidden Layer {idx} (Stage {idx+1})"

            col = cols[idx % len(cols)]
            with col:
                st.markdown(f"""
                <div style="border-left: 4px solid {group.color}; padding: 0.6rem 0.8rem; background: #F8FAFC; border-radius: 6px; margin-bottom: 0.5rem; border: 1px solid #E2E8F0;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase;">{stage_badge}</div>
                    <strong style="color: #1E293B; font-size: 1.0rem;">{group.name}</strong><br>
                    <span style="font-size: 0.8rem; color: #64748B;">Neurons / Machines:</span> <strong>{[f'M{m}' for m in group.machine_ids]}</strong><br>
                    <span style="font-size: 0.8rem; color: #64748B;">Parallel Capacity:</span> {len(group.machine_ids)} units
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# Tab 5: Workload & Validation (Pre-Simulation)
# ==============================================================================
with tab_inspect:
    st.subheader("Static Visual Inspection & Validation")
    st.caption("Review machine workload balance and routing health before exporting. No simulation execution or Gantt chart is generated here.")

    # Validation banner
    is_valid, errors, warnings = validate_scenario(scenario)
    if is_valid:
        st.success("✅ Scenario Integrity Check Passed: Ready for simulation scripts!")
    else:
        for err in errors:
            st.error(f"❌ Error: {err}")
    for warn in warnings:
        st.warning(f"⚠️ Warning: {warn}")

    st.markdown("---")

    # Workload statistics
    workload_stats = compute_workload_distribution(scenario)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Scheduled Work", f"{workload_stats['total_workload']} s")
    col_m2.metric("Bottleneck Machine", f"Machine {workload_stats['bottleneck_machine']}")
    col_m3.metric("Peak Machine Load", f"{workload_stats['bottleneck_workload']} s")

    # Plot machine workload distribution
    st.markdown("#### 📊 Planned Workload per Machine")
    m_data = workload_stats["machine_workload"]
    df_machines = pd.DataFrame({
        "Machine": [f"Machine {m}" for m in m_data.keys()],
        "Total Processing Time (s)": list(m_data.values())
    })

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#EF4444" if m == workload_stats["bottleneck_machine"] else "#3B82F6" for m in m_data.keys()]
    ax.bar(df_machines["Machine"], df_machines["Total Processing Time (s)"], color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Processing Seconds", fontsize=11)
    ax.set_title("Machine Workload Distribution (Red = Bottleneck)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Routing Journey Table
    st.markdown("#### 🗺️ Job Process Journey")
    journey_rows = []
    for job in scenario.jobs:
        if scenario.mode == "simplified":
            steps_str = " ➔ ".join([f"M{s.machine_id} ({s.duration}s)" for s in job.steps])
        else:
            steps_str = " ➔ ".join([f"[{s.group_name}] ({s.duration}s)" if s.group_name else f"M{s.machine_id} ({s.duration}s)" for s in job.steps])
        journey_rows.append({
            "Job": job.name,
            "Total Steps": len(job.steps),
            "Total Duration": f"{sum(s.duration for s in job.steps):.1f} s",
            "Process Route": steps_str
        })
    st.dataframe(pd.DataFrame(journey_rows), use_container_width=True)


# ==============================================================================
# Tab 6: Export & Code Snippets
# ==============================================================================
with tab_export:
    st.subheader("Parameter Export & Integration Snippets")
    st.caption("Download the configured parameters as a ready-to-use Python module or JSON file, or copy the code snippet directly into your numerical scripts.")

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.download_button(
            label="📥 Download scenario_parameters.py",
            data=export_to_python_code(scenario),
            file_name="scenario_parameters.py",
            mime="text/x-python",
            type="primary",
            use_container_width=True
        )
    with col_ex2:
        st.download_button(
            label="📥 Download scenario.json",
            data=export_to_json(scenario),
            file_name="scenario.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("#### 📋 Preview: `scenario_parameters.py`")
    st.code(export_to_python_code(scenario), language="python")

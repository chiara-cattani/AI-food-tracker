import time
import json
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta

from db import (
    init_db, register_user, authenticate_user,
    save_meal_session, get_meal_sessions, get_all_meal_items_flat,
    save_food_correction, get_food_corrections,
    get_daily_nutrition,
)
from vision import recognize_food
from food_api import search_nutrition
from utils import (
    compute_nutrition, safe_grams, unit_to_grams, image_hash,
    compress_image, classify_meal_time, UNIT_OPTIONS,
    CONFIDENCE_THRESHOLD,
)

# ═════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + CSS
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🍝 AI Food Tracker",
    page_icon="🍝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top:1rem!important;padding-bottom:1rem!important;
    max-width:720px!important;margin:0 auto}
.stButton>button {width:100%;padding:.75rem 1rem;font-size:1.1rem;
    border-radius:12px;margin-top:.25rem;margin-bottom:.25rem}
.food-card {background:#fff;border-radius:14px;padding:1rem 1.2rem;
    margin-bottom:.75rem;border:1px solid #d0d0d0;color:#1a1a1a!important}
.food-card h3,.food-card p,.food-card strong,.food-card span
    {color:#1a1a1a!important}
.food-card p {margin:.15rem 0;font-size:.95rem}
.food-card h3 {margin:0 0 .4rem 0;font-size:1.15rem}
.summary-card {background:#e8f5e9;border-radius:14px;padding:1rem 1.2rem;
    margin-bottom:.75rem;border:1px solid #66bb6a;color:#1a1a1a!important}
.summary-card h3,.summary-card p,.summary-card strong
    {color:#1a1a1a!important}
.summary-card p {margin:.15rem 0;font-size:.95rem}
.summary-card h3 {margin:0 0 .4rem 0;font-size:1.15rem}
.section-title {font-size:1.3rem;font-weight:700;margin-top:1.5rem;
    margin-bottom:.5rem}
.stCameraInput>div {width:100%!important}
.chip {display:inline-block;padding:2px 8px;border-radius:10px;
    font-size:.75rem;font-weight:600;margin-right:4px}
.chip-ai {background:#e3f2fd;color:#1565c0}
.chip-edited {background:#fff3e0;color:#e65100}
.chip-manual {background:#f3e5f5;color:#7b1fa2}
.chip-low {background:#ffebee;color:#c62828}
.chip-off {background:#e8f5e9;color:#2e7d32}
.chip-fb {background:#fff8e1;color:#f57f17}
.chip-mig {background:#eceff1;color:#546e6a}
.hist-card {background:#fafafa;border-radius:12px;padding:.8rem 1rem;
    margin-bottom:.5rem;border:1px solid #e0e0e0;color:#1a1a1a!important}
.hist-card p,.hist-card strong,.hist-card span {color:#1a1a1a!important}
</style>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  INIT
# ═════════════════════════════════════════════════════════════════════════════
init_db()

_DEFAULTS = {
    "user_id": None, "username": None,
    "results": None, "image_bytes": None, "image_hash": None,
    "uploaded_at": None, "ai_raw_json": None, "source": None,
    "deleted_item": None, "login_attempts": {},
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═════════════════════════════════════════════════════════════════════════════
#  LOGIN / REGISTER
# ═════════════════════════════════════════════════════════════════════════════
_MAX_ATTEMPTS = 5
_LOCKOUT_SEC = 300


def _show_login():
    st.markdown("# 🍝 AI Food Tracker")
    st.caption("Track your meals with AI-powered food recognition.")
    login_tab, reg_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            sub = st.form_submit_button(
                "Log in", type="primary", use_container_width=True
            )
        if sub:
            if not u or not p:
                st.warning("Please enter both username and password.")
            else:
                attempts = st.session_state.login_attempts.get(u, [])
                cutoff = time.time() - _LOCKOUT_SEC
                recent = [t for t in attempts if t > cutoff]
                st.session_state.login_attempts[u] = recent
                if len(recent) >= _MAX_ATTEMPTS:
                    st.error("Too many failed attempts. Try again in a few minutes.")
                else:
                    uid = authenticate_user(u, p)
                    if uid:
                        st.session_state.user_id = uid
                        st.session_state.username = u
                        st.session_state.login_attempts.pop(u, None)
                        st.rerun()
                    else:
                        recent.append(time.time())
                        st.session_state.login_attempts[u] = recent
                        st.error("Invalid username or password.")

    with reg_tab:
        with st.form("register_form"):
            nu = st.text_input("Choose a username")
            np_ = st.text_input("Choose a password", type="password")
            np2 = st.text_input("Confirm password", type="password")
            rs = st.form_submit_button("Create account", use_container_width=True)
        if rs:
            if not nu or not np_:
                st.warning("Please fill in all fields.")
            elif np_ != np2:
                st.error("Passwords do not match.")
            elif len(np_) < 4:
                st.warning("Minimum 4 characters.")
            else:
                if register_user(nu, np_):
                    st.success("✅ Account created! Switch to **Log in**.")
                else:
                    st.error("Username already taken.")


if st.session_state.user_id is None:
    _show_login()
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# 🍝 AI Food Tracker")
cw, cl = st.columns([3, 1])
with cw:
    st.caption(f"Welcome, **{st.session_state.username}**")
with cl:
    if st.button("Logout", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Onboarding ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown(
        "1. **📸 Take a photo** of your meal (top-down, good lighting)\n"
        "2. **🔍 Analyze** — AI identifies foods and portions\n"
        "3. **✏️ Edit** — correct names, adjust grams, remove / add items\n"
        "4. **💾 Save** — stored in your personal history\n"
        "5. **📊 Track** — view trends, export data"
    )

# ═════════════════════════════════════════════════════════════════════════════
#  📸  CAMERA
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-title">📸 Capture your meal</p>', unsafe_allow_html=True
)
cam_tab, up_tab = st.tabs(["Camera", "Upload"])
with cam_tab:
    camera_img = st.camera_input("Take a picture of your food")
with up_tab:
    upload_img = st.file_uploader(
        "Or upload a photo", type=["jpg", "jpeg", "png", "webp"]
    )

img_src = camera_img or upload_img
if img_src:
    new_bytes = img_src.getvalue()
    new_h = image_hash(new_bytes)
    if new_h != st.session_state.image_hash:
        st.session_state.image_bytes = new_bytes
        st.session_state.image_hash = new_h
        st.session_state.uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.results = None
        st.session_state.ai_raw_json = None
        st.session_state.source = "camera" if camera_img else "upload"
    st.image(
        st.session_state.image_bytes, caption="Your meal", use_container_width=True
    )

# ═════════════════════════════════════════════════════════════════════════════
#  ACTION BUTTONS
# ═════════════════════════════════════════════════════════════════════════════
b1, b2, b3 = st.columns(3)
with b1:
    analyze_clicked = st.button(
        "🔍 Analyze", type="primary", use_container_width=True
    )
with b2:
    manual_start = st.button("📝 Manual meal", use_container_width=True)
with b3:
    if st.button("🗑️ Clear", use_container_width=True):
        for k in [
            "results", "image_bytes", "image_hash",
            "uploaded_at", "ai_raw_json", "source", "deleted_item",
        ]:
            st.session_state[k] = None
        st.rerun()

if manual_start and st.session_state.results is None:
    st.session_state.results = []
    st.session_state.source = "manual"
    st.rerun()

# ── Analyze ──────────────────────────────────────────────────────────────────
if analyze_clicked:
    if st.session_state.image_bytes is None:
        st.warning("⚠️ Take or upload a photo first.")
    else:
        try:
            compressed = compress_image(st.session_state.image_bytes)
            with st.spinner("🧠 Recognizing food…"):
                recognition, raw_json = recognize_food(compressed)
            st.session_state.ai_raw_json = raw_json
            foods = recognition.get("foods", [])
            if not foods:
                st.warning("No food detected. Try a clearer photo.")
                st.session_state.results = None
            else:
                enriched: list[dict] = []
                with st.spinner("🍽️ Fetching nutrition…"):
                    for item in foods:
                        name = item.get("name", "Unknown")
                        conf = item.get("confidence", 0)
                        grams = safe_grams(item.get("estimated_grams"))
                        n100, matched, src = search_nutrition(name)
                        nutr = compute_nutrition(n100, grams)
                        enriched.append(
                            {
                                "name": name,
                                "original_name": name,
                                "confidence": conf,
                                "grams": grams,
                                "entered_unit": "grams",
                                "entered_quantity": grams,
                                "nutrition_source": src,
                                "matched_product_name": matched,
                                "status": "ai_detected",
                                **nutr,
                            }
                        )
                st.session_state.results = enriched
                if not st.session_state.uploaded_at:
                    st.session_state.uploaded_at = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
        except ValueError as ve:
            st.error(f"⚙️ {ve}")
        except Exception as exc:
            st.error(f"❌ {exc}")
            st.info("Try again with a different photo.")


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def _recalc(idx: int, new_name: str, new_grams: float):
    item = st.session_state.results[idx]
    old_name = item["name"]
    if (
        item.get("original_name")
        and new_name != old_name
        and item["status"] in ("ai_detected", "edited")
    ):
        save_food_correction(
            st.session_state.user_id, item["original_name"], new_name
        )
    n100, matched, src = search_nutrition(new_name)
    nutr = compute_nutrition(n100, new_grams)
    if item["status"] == "ai_detected" and (
        new_name != old_name or new_grams != item["grams"]
    ):
        item["status"] = "edited"
    item.update(
        {
            "name": new_name,
            "grams": new_grams,
            "nutrition_source": src,
            "matched_product_name": matched,
            **nutr,
        }
    )


def _status_chip(status: str) -> str:
    m = {
        "ai_detected": ("AI ✓", "chip-ai"),
        "edited": ("Edited", "chip-edited"),
        "manually_added": ("Manual", "chip-manual"),
        "migrated": ("Migrated", "chip-mig"),
    }
    label, cls = m.get(status, (status, "chip-ai"))
    return f'<span class="chip {cls}">{label}</span>'


def _source_chip(src: str) -> str:
    if src == "openfoodfacts":
        return '<span class="chip chip-off">OpenFoodFacts</span>'
    return '<span class="chip chip-fb">Fallback ⚠️</span>'


# ═════════════════════════════════════════════════════════════════════════════
#  📊  RESULTS / EDIT
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.results is not None:
    results = st.session_state.results

    if len(results) > 0:
        st.markdown(
            '<p class="section-title">📊 Results</p>', unsafe_allow_html=True
        )
        st.caption("Edit name or grams → **Update**. Press ❌ to remove.")

        # Undo delete
        if st.session_state.deleted_item is not None:
            d_idx, d_item = st.session_state.deleted_item
            if st.button("↩️ Undo delete", use_container_width=True):
                results.insert(min(d_idx, len(results)), d_item)
                st.session_state.deleted_item = None
                st.rerun()

        to_delete = None
        for idx, food in enumerate(results):
            conf_pct = int(food.get("confidence", 0) * 100)

            # ── Edit row ─────────────────────────────────────────────────
            cn, cg, cu, cd = st.columns([3, 1.5, 1, 1])
            with cn:
                new_name = st.text_input(
                    "Food",
                    value=food["name"],
                    key=f"n_{idx}",
                    label_visibility="collapsed",
                )
            with cg:
                new_grams = st.number_input(
                    "g",
                    value=float(food["grams"]),
                    min_value=1.0,
                    step=10.0,
                    key=f"g_{idx}",
                    label_visibility="collapsed",
                )
            with cu:
                if st.button(
                    "🔄", key=f"u_{idx}", help="Update", use_container_width=True
                ):
                    _recalc(idx, new_name, new_grams)
                    st.rerun()
            with cd:
                if st.button(
                    "❌", key=f"d_{idx}", help="Delete", use_container_width=True
                ):
                    to_delete = idx

            # ── Quick grams adjust + duplicate ───────────────────────────
            qa1, qa2, qa3, qa4, qa5 = st.columns(5)
            for col, delta in [(qa1, -50), (qa2, -10), (qa3, 10), (qa4, 50)]:
                with col:
                    if st.button(
                        f"{delta:+d}g",
                        key=f"q_{idx}_{delta}",
                        use_container_width=True,
                    ):
                        new_g = max(1.0, food["grams"] + delta)
                        _recalc(idx, food["name"], new_g)
                        st.rerun()
            with qa5:
                if st.button(
                    "📋", key=f"cp_{idx}", help="Duplicate", use_container_width=True
                ):
                    dup = dict(food)
                    dup["status"] = "manually_added"
                    dup["confidence"] = 0
                    results.insert(idx + 1, dup)
                    st.rerun()

            # ── Info card ────────────────────────────────────────────────
            chips = _status_chip(food["status"])
            chips += _source_chip(food.get("nutrition_source", "fallback"))
            if (
                food.get("confidence", 1) < CONFIDENCE_THRESHOLD
                and food["status"] == "ai_detected"
            ):
                chips += '<span class="chip chip-low">Low confidence</span>'
            matched = food.get("matched_product_name")
            matched_html = (
                f"<br><span style='font-size:.8rem;color:#666'>"
                f"Matched: {matched}</span>"
                if matched
                else ""
            )
            st.markdown(
                f"""<div class="food-card"><p>{chips}</p>
<p>🎯 <strong>{conf_pct}%</strong> &nbsp;|&nbsp;
⚖️ <strong>{food['grams']:.0f} g</strong></p>
<p>🔥 <strong>{food['calories']:.0f}</strong> kcal &nbsp;|&nbsp;
🥩 <strong>{food['protein']:.1f} g</strong> &nbsp;|&nbsp;
🧈 <strong>{food['fat']:.1f} g</strong> &nbsp;|&nbsp;
🍞 <strong>{food['carbs']:.1f} g</strong></p>{matched_html}</div>""",
                unsafe_allow_html=True,
            )

        if to_delete is not None:
            st.session_state.deleted_item = (to_delete, results.pop(to_delete))
            st.rerun()

    # ═════════════════════════════════════════════════════════════════════
    #  ➕  MANUAL ADD
    # ═════════════════════════════════════════════════════════════════════
    st.markdown(
        '<p class="section-title">➕ Add food manually</p>', unsafe_allow_html=True
    )
    add_name = st.text_input(
        "Food name", key="add_name", placeholder="e.g. banana"
    )
    ac1, ac2, ac3 = st.columns([1.2, 1, 1])
    with ac1:
        add_unit = st.selectbox("Unit", UNIT_OPTIONS, key="add_unit")
    with ac2:
        default_qty = 150.0 if add_unit == "grams" else 1.0
        add_qty = st.number_input(
            "Qty",
            value=default_qty,
            min_value=0.1,
            step=1.0 if add_unit != "grams" else 10.0,
            key="add_qty",
        )
    with ac3:
        st.markdown("<br>", unsafe_allow_html=True)
        add_btn = st.button("➕ Add", key="add_btn", use_container_width=True)

    if add_btn and add_name.strip():
        grams = unit_to_grams(add_unit, add_qty)
        n100, matched, src = search_nutrition(add_name.strip())
        nutr = compute_nutrition(n100, grams)
        results.append(
            {
                "name": add_name.strip(),
                "original_name": add_name.strip(),
                "confidence": 0,
                "grams": grams,
                "entered_unit": add_unit,
                "entered_quantity": add_qty,
                "nutrition_source": src,
                "matched_product_name": matched,
                "status": "manually_added",
                **nutr,
            }
        )
        st.rerun()

    # ═════════════════════════════════════════════════════════════════════
    #  📋  MEAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    if len(results) > 0:
        tot_c = sum(f["calories"] for f in results)
        tot_p = sum(f["protein"] for f in results)
        tot_f = sum(f["fat"] for f in results)
        tot_cb = sum(f["carbs"] for f in results)
        st.markdown(
            f"""<div class="summary-card">
<h3>📋 Meal Summary — {len(results)} item{"s" if len(results) != 1 else ""}</h3>
<p>🔥 <strong>{tot_c:.0f}</strong> kcal &nbsp;|&nbsp;
🥩 <strong>{tot_p:.1f} g</strong> protein &nbsp;|&nbsp;
🧈 <strong>{tot_f:.1f} g</strong> fat &nbsp;|&nbsp;
🍞 <strong>{tot_cb:.1f} g</strong> carbs</p></div>""",
            unsafe_allow_html=True,
        )

        # ═════════════════════════════════════════════════════════════════
        #  💾  SAVE
        # ═════════════════════════════════════════════════════════════════
        st.markdown(
            '<p class="section-title">💾 Save meal</p>', unsafe_allow_html=True
        )
        now = datetime.now()
        sc1, sc2 = st.columns(2)
        with sc1:
            eaten_date = st.date_input(
                "Eaten on", value=now.date(), key="eaten_date"
            )
        with sc2:
            eaten_time = st.time_input(
                "At time",
                value=now.time().replace(second=0, microsecond=0),
                key="eaten_time",
            )

        if st.button("💾 Save meal", type="primary", use_container_width=True):
            uploaded_at = st.session_state.uploaded_at
            eaten_at = datetime.combine(eaten_date, eaten_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            source = st.session_state.source or "manual"
            save_meal_session(
                user_id=st.session_state.user_id,
                uploaded_at=uploaded_at,
                eaten_at=eaten_at,
                source=source,
                image_hash=st.session_state.image_hash,
                ai_raw_json=st.session_state.ai_raw_json,
                items=results,
            )
            st.success("✅ Meal saved correctly!")
    elif len(results) == 0:
        st.info("No items yet. Use **Analyze** or **Add food manually** above.")

# ═════════════════════════════════════════════════════════════════════════════
#  📜  HISTORY
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p class="section-title">📜 Meal History</p>', unsafe_allow_html=True
)

hc1, hc2, hc3 = st.columns(3)
with hc1:
    h_from = st.date_input(
        "From", value=date.today() - timedelta(days=30), key="h_from"
    )
with hc2:
    h_to = st.date_input("To", value=date.today(), key="h_to")
with hc3:
    h_search = st.text_input("Search food", key="h_search", placeholder="e.g. chicken")

sessions = get_meal_sessions(
    st.session_state.user_id,
    date_from=h_from.strftime("%Y-%m-%d") if h_from else None,
    date_to=h_to.strftime("%Y-%m-%d") if h_to else None,
    food_search=h_search.strip() or None,
)

if sessions:
    for sess in sessions:
        items = sess.get("items", [])
        total_kcal = sum(i["calories"] for i in items)
        total_p = sum(i["protein"] for i in items)
        total_f = sum(i["fat"] for i in items)
        total_c = sum(i["carbs"] for i in items)
        eaten = sess.get("eaten_at", "")
        source = sess.get("source", "")

        try:
            hour = datetime.strptime(eaten, "%Y-%m-%d %H:%M:%S").hour
            meal_type = classify_meal_time(hour)
        except ValueError:
            meal_type = ""

        header = (
            f"🕐 {eaten} — {meal_type} ({source}) — "
            f"🔥 {total_kcal:.0f} kcal"
        )
        with st.expander(header, expanded=False):
            for it in items:
                status = it.get("status", "")
                st.markdown(
                    f"""<div class="hist-card">
<p><strong>{it['food_name']}</strong> — {it['grams']:.0f}g</p>
<p>🔥 {it['calories']:.0f} kcal | 🥩 {it['protein']:.1f}g
| 🧈 {it['fat']:.1f}g | 🍞 {it['carbs']:.1f}g
<span style="float:right;font-size:.8rem;color:#888">{status}</span></p>
</div>""",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"**Total:** {total_kcal:.0f} kcal | "
                f"P: {total_p:.1f}g | F: {total_f:.1f}g | C: {total_c:.1f}g"
            )
else:
    st.info("No meals found. Start tracking above! 🍽️")

# ═════════════════════════════════════════════════════════════════════════════
#  📊  ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p class="section-title">📊 Analytics</p>', unsafe_allow_html=True
)

daily = get_daily_nutrition(st.session_state.user_id)
if daily:
    df_daily = pd.DataFrame(daily)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.sort_values("date")

    st.markdown("**Daily Calories (last 30 days)**")
    chart_data = df_daily.set_index("date")[["total_calories"]].rename(
        columns={"total_calories": "Calories"}
    )
    st.bar_chart(chart_data)

    st.markdown("**Daily Macros**")
    macro_data = df_daily.set_index("date")[
        ["total_protein", "total_fat", "total_carbs"]
    ]
    macro_data.columns = ["Protein (g)", "Fat (g)", "Carbs (g)"]
    st.line_chart(macro_data)

    corrections = get_food_corrections(st.session_state.user_id)
    if corrections:
        st.markdown("**Food Name Corrections**")
        df_corr = pd.DataFrame(corrections)
        df_corr.columns = ["Original", "Corrected to", "Times"]
        st.dataframe(df_corr, use_container_width=True, hide_index=True)
else:
    st.info("Save some meals to see analytics.")

# ═════════════════════════════════════════════════════════════════════════════
#  📥  EXPORT
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p class="section-title">📥 Export Data</p>', unsafe_allow_html=True
)

all_items = get_all_meal_items_flat(st.session_state.user_id)
if all_items:
    df_export = pd.DataFrame(all_items)
    ec1, ec2 = st.columns(2)
    with ec1:
        csv_data = df_export.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            csv_data,
            "meal_data.csv",
            "text/csv",
            use_container_width=True,
        )
    with ec2:
        json_data = df_export.to_json(orient="records", indent=2)
        st.download_button(
            "📋 Download JSON",
            json_data,
            "meal_data.json",
            "application/json",
            use_container_width=True,
        )
else:
    st.info("No data to export yet.")

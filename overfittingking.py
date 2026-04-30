        out = final_oos_strategy_returns(
            train_df,
            test_df,
            best_win,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
        )
        raw_rets = out["raw"]
        filtered_rets = out["filtered"]
    else:
        out = compute_strategy_returns(
            train_df,
            best_win,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
        )
        raw_rets = out["raw"]
        filtered_rets = out["filtered"]

    raw_sharpe = _safe_sharpe(raw_rets)
    filtered_sharpe = _safe_sharpe(filtered_rets)
    filtered_equity = np.cumprod(1.0 + np.concatenate(([0.0], filtered_rets)))

    benchmark_prices = test_df["close"].to_numpy(dtype=float) if use_oos else train_df["close"].to_numpy(dtype=float)
    buy_hold_sharpe = _safe_sharpe(buy_and_hold_returns(benchmark_prices))
    ma_sharpe = _safe_sharpe(moving_average_crossover_returns(benchmark_prices, tc_bps))

    rng = np.random.default_rng(12345)
    progress_bar = st.progress(0, text="Running bootstrap under the null...")
    best_boot = np.empty(n_boot, dtype=float)
    train_prices = train_df["close"].to_numpy(dtype=float)
    for i in range(n_boot):
        if bootstrap_mode == "Block bootstrap":
            boot_prices = prices_from_block_bootstrap(train_prices, rng, block_size)
        else:
            boot_prices = prices_from_resampled_returns(train_prices, rng)
        boot_feat = feature_df_from_prices(boot_prices)
        boot_best_win, boot_scores = choose_best_window(
            boot_feat,
            candidate_windows,
            tc_bps,
            use_vwap_filter,
            use_ema_filter,
            use_volume_filter,
            min_volume_z,
        )
        best_boot[i] = float(boot_scores.iloc[0]["walk_forward_sharpe"]) if boot_best_win is not None else np.nan
        progress_bar.progress((i + 1) / n_boot)
    progress_bar.empty()

    valid_boot = best_boot[~np.isnan(best_boot)]
    p_value = float(np.mean(valid_boot >= train_cv_sharpe)) if valid_boot.size else np.nan
    q95 = float(np.quantile(valid_boot, 0.95)) if valid_boot.size else np.nan

    cross_asset_rows: List[Dict[str, object]] = []
    if run_cross_asset and use_oos:
        with st.spinner("Running cross-asset validation..."):
            for symbol in SECONDARY_SYMBOLS:
                cross_asset_rows.append(
                    evaluate_cross_asset(
                        target_bars,
                        symbol,
                        best_win,
                        tc_bps,
                        oos_pct,
                        use_vwap_filter,
                        use_ema_filter,
                        use_volume_filter,
                        min_volume_z,
                    )
                )

    report_checks = [
        ("Train CV > bootstrap 95%", bool(not np.isnan(q95) and train_cv_sharpe > q95)),
        ("Filtered holdout Sharpe > 0", bool(not np.isnan(filtered_sharpe) and filtered_sharpe > 0)),
        ("Filtered > raw wavelet", bool(not np.isnan(filtered_sharpe) and not np.isnan(raw_sharpe) and filtered_sharpe > raw_sharpe)),
        ("Filtered > buy-and-hold", bool(not np.isnan(filtered_sharpe) and not np.isnan(buy_hold_sharpe) and filtered_sharpe > buy_hold_sharpe)),
        ("Filtered > MA baseline", bool(not np.isnan(filtered_sharpe) and not np.isnan(ma_sharpe) and filtered_sharpe > ma_sharpe)),
    ]
    if cross_asset_rows:
        cross_asset_ok = any(
            row.get("status") == "ok"
            and not np.isnan(row.get("holdout_sharpe", np.nan))
            and row.get("holdout_sharpe", np.nan) > 0
            for row in cross_asset_rows
        )
        report_checks.append(("Positive on secondary asset", cross_asset_ok))

    passed_checks = sum(1 for _, ok in report_checks if ok)
    total_checks = len(report_checks)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected window", f"{best_win} bars")
        st.metric("Train CV Sharpe", f"{train_cv_sharpe:.4f}")
    with col2:
        st.metric("Raw wavelet Sharpe", f"{raw_sharpe:.4f}")
        st.metric("Filtered Sharpe", f"{filtered_sharpe:.4f}")
    with col3:
        st.metric("Buy/Hold Sharpe", f"{buy_hold_sharpe:.4f}")
        st.metric("MA Sharpe", f"{ma_sharpe:.4f}")
    with col4:
        st.metric("Checks passed", f"{passed_checks}/{total_checks}")
        st.metric("Bootstrap p-value", "n/a" if np.isnan(p_value) else f"{p_value:.1%}")

    st.subheader("Pass / Fail Report")
    report_df = pd.DataFrame(
        [{"check": label, "result": "PASS" if ok else "FAIL"} for label, ok in report_checks]
    )
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.subheader("Window Search Results")
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.subheader("Strategy Comparison")
    compare_df = pd.DataFrame(
        [
            {"strategy": "Raw wavelet", "sharpe": raw_sharpe},
            {"strategy": "Feature-filtered wavelet", "sharpe": filtered_sharpe},
            {"strategy": "Buy and hold", "sharpe": buy_hold_sharpe},
            {"strategy": "24/72 MA crossover", "sharpe": ma_sharpe},
        ]
    )
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    filter_state_df = pd.DataFrame(
        [
            {"filter": "VWAP confirmation", "enabled": use_vwap_filter},
            {"filter": "EMA slope confirmation", "enabled": use_ema_filter},
            {"filter": "Volume z-score", "enabled": use_volume_filter},
            {"filter": "Min volume z", "enabled": min_volume_z},
        ]
    )
    st.subheader("Filter Configuration")
    st.dataframe(filter_state_df, use_container_width=True, hide_index=True)

    if cross_asset_rows:
        st.subheader("Cross-Asset Check")
        st.dataframe(pd.DataFrame(cross_asset_rows), use_container_width=True, hide_index=True)

    st.subheader("Filtered Equity Curve")
    chart_label = "Holdout Equity" if use_oos else "In-sample Equity"
    st.line_chart(pd.DataFrame({chart_label: filtered_equity}), use_container_width=True)

    st.caption(
        "Candles and live feature snapshots are stored in SQLite. Historical backtests currently use candle-derived features only; live orderbook imbalance is logged for future out-of-sample research."
    )
else:
    st.info("Click the button to fetch real BTC data and run the feature-filtered validation pass.")

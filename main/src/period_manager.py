import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d


class PeriodManager:
    def __init__(self, df, periods_df, experiment_id, offset):
        
        self.df = df
        self.experiment_id = experiment_id
        self.time_offset = offset

        self.periods_df = periods_df.copy()
        self.transition_df = pd.DataFrame()
        self.final_period_df = pd.DataFrame()
        self._synchronize_periods()

    def _synchronize_periods(self):
        print(
            f"🔄 Synchronizing periods for {self.experiment_id} (offset: {self.time_offset}h)..."
        )

        day_start = self.df["datetime"].iloc[0].replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - pd.Timedelta(hours=self.time_offset)
        reference_unix_time = day_start.timestamp()

        self.periods_df["start_unix"] = (
            self.periods_df["start"] * 20 + reference_unix_time
        )
        self.periods_df["end_unix"] = self.periods_df["end"] * 20 + reference_unix_time

        self.periods_df["start_unix_datetime"] = pd.to_datetime(
            self.periods_df["start_unix"], unit="s"
        )
        self.periods_df["end_unix_datetime"] = pd.to_datetime(
            self.periods_df["end_unix"], unit="s"
        )

    def detect_transitions(
        self,
        threshold_pos=0.2,
        threshold_neg=-0.2,
        smooth_sigma=3,
        freq="20s",
        debug=False,
    ):
        print(f"🔍 Detecting transitions for {self.experiment_id}...")

        # === Compute Activity & Derivatives ===
        activity_series = self.df.set_index("datetime").resample(freq).size()
        time_seconds = (
            activity_series.index - activity_series.index[0]
        ).total_seconds()

        smoothed = gaussian_filter1d(activity_series.values, sigma=smooth_sigma)
        derivative = np.gradient(smoothed, time_seconds)
        norm_derivative = derivative / (np.max(np.abs(derivative)) or 1.0)

        # === Transition Detection ===
        trans_df = pd.DataFrame(
            {
                "smoothed": smoothed,
                "norm_derivative": norm_derivative,
                "is_up": norm_derivative > threshold_pos,
                "is_down": norm_derivative < threshold_neg,
            },
            index=activity_series.index,
        )

        trans_df["is_transition"] = trans_df["is_up"] | trans_df["is_down"]
        trans_df["group"] = (
            trans_df["is_transition"] != trans_df["is_transition"].shift()
        ).cumsum()

        # === Extract Transitions
        self.transition_df = self._extract_transitions(trans_df)
        print(f"✅ Detected {len(self.transition_df)} transitions.")

        # === Adjust Final Periods ===
        self._adjust_periods()
        print(f"✅ Final periods updated: {len(self.final_period_df)} entries.")
        

        if debug:
            return {
                "activity_series":  activity_series,
                "smoothed": smoothed,
                "norm_derivative": norm_derivative,
                "transition_df": self.transition_df,
                "smooth_sigma": smooth_sigma,
            }

        return None

    def _extract_transitions(self, trans_df):
        periods = []
        up_idx, down_idx = 1, 1

        for _, group_df in trans_df[trans_df["is_transition"]].groupby("group"):
            if len(group_df) > 1:
                times = group_df.index
                if group_df["is_up"].all():
                    label, typ = f"T_UP_{up_idx}", "t_up"
                    up_idx += 1
                elif group_df["is_down"].all():
                    label, typ = f"T_DOWN_{down_idx}", "t_down"
                    down_idx += 1

                periods.append(
                    {
                        "start_unix_datetime": times[0],
                        "end_unix_datetime": times[-1],
                        "label": label,
                        "type": typ,
                    }
                )

        return pd.DataFrame(periods)

    def _adjust_periods(self):
        print(f"🛠️ Adjusting periods with transitions for {self.experiment_id}...")

        adjusted_periods = []
        seen_transitions = set()

        for _, period in self.periods_df.iterrows():
            start, end = period["start_unix_datetime"], period["end_unix_datetime"]
            overlapping = self.transition_df[
                (self.transition_df["start_unix_datetime"] < end)
                & (self.transition_df["end_unix_datetime"] > start)
            ]

            overlapping = overlapping[
                ~(
                    (overlapping["start_unix_datetime"] >= start)
                    & (overlapping["end_unix_datetime"] <= end)
                )
            ].sort_values("start_unix_datetime")

            last_pointer = start

            if overlapping.empty:
                adjusted_periods.append(period)
                continue

            for _, trans in overlapping.iterrows():
                tr_start, tr_end = (
                    trans["start_unix_datetime"],
                    trans["end_unix_datetime"],
                )
                tr_id = (tr_start, tr_end)

                if last_pointer < tr_start:
                    pre_row = period.copy()
                    pre_row["start_unix_datetime"] = last_pointer
                    pre_row["end_unix_datetime"] = min(tr_start, end)
                    adjusted_periods.append(pre_row)

                if tr_start < end and tr_id not in seen_transitions:
                    adjusted_periods.append(trans)
                    seen_transitions.add(tr_id)

                last_pointer = max(last_pointer, tr_end)

            if last_pointer < end:
                post_row = period.copy()
                post_row["start_unix_datetime"] = last_pointer
                post_row["end_unix_datetime"] = end
                adjusted_periods.append(post_row)

        result_df = pd.DataFrame(adjusted_periods)

        result_df["start_unix"] = (
            result_df["start_unix_datetime"].astype("int64") // 10**9
        )
        result_df["end_unix"] = result_df["end_unix_datetime"].astype("int64") // 10**9

        # Re-label transitions cleanly
        for typ, prefix in [("t_down", "Down_Trans"), ("t_up", "Up_Trans")]:
            mask = result_df["type"] == typ
            result_df.loc[mask, "label"] = [
                f"{prefix}_{i+1}" for i in range(mask.sum())
            ]

        self.final_period_df = result_df.sort_values("start_unix_datetime").reset_index(
            drop=True
        )

    def get_final_periods(self):
        return self.final_period_df
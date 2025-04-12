import datetime as dt

import numpy as np
import polars as pl


class Trial:
    """"""

    def __init__(self):
        """"""
        # np.random.seed(101)
        np.random.seed()

    def build(self):
        """"""
        n_day = 5
        # tick_duration_sec = 60*60
        tick_duration_sec = 60
        n_tick_per_day = int(7 * 3600 / tick_duration_sec)
        day_start_hour = 9
        n_bd_year = 252

        self.n_tick_per_day = n_tick_per_day

        start_date = dt.datetime(2025, 4, 8)
        dates = []
        for day in range(n_day):
            for i in range(n_tick_per_day):
                time_delta = dt.timedelta(
                    days=day, hours=day_start_hour, seconds=i * tick_duration_sec
                )
                dates.append(start_date + time_delta)

        n = len(dates)
        spot_A_start = 100
        spot_B_start = 100

        vol_A = 0.20
        vol_B = 0.40

        self.vol_A = vol_A
        self.vol_B = vol_B

        tick_vol_A = vol_A / np.sqrt(n_bd_year * n_tick_per_day)
        tick_vol_B = vol_B / np.sqrt(n_bd_year * n_tick_per_day)

        # random returns
        returns_A = np.random.normal(0, tick_vol_A, n)
        returns_B = np.random.normal(0, tick_vol_B, n)

        # Calculate spot prices
        spot_A = [spot_A_start]
        spot_B = [spot_B_start]

        for i in range(1, n):
            spot_A.append(spot_A[-1] * (1 + returns_A[i]))
            spot_B.append(spot_B[-1] * (1 + returns_B[i]))

        df = (
            pl.DataFrame(
                {
                    "date": pl.Series(dates, dtype=pl.Datetime),
                    "spot_A": pl.Series(spot_A, dtype=pl.Float64),
                    "spot_B": pl.Series(spot_B, dtype=pl.Float64),
                }
            )
            .with_columns(
                ret_A=(pl.col("spot_A") / pl.col("spot_A").shift(+1) - 1),
                ret_B=(pl.col("spot_B") / pl.col("spot_B").shift(+1) - 1),
                log_ret_A=(pl.col("spot_A") / pl.col("spot_A").shift(+1)).log(),
                log_ret_B=(pl.col("spot_B") / pl.col("spot_B").shift(+1)).log(),
            )
            .with_columns(
                ret_sq_A=pl.col("ret_A").fill_null(0.0).pow(2),
                ret_sq_B=pl.col("ret_B").fill_null(0.0).pow(2),
            )
        )

        self.df1 = df

        df = (
            df.unpivot(
                index=["date"],
                on=["ret_sq_A", "ret_sq_B"],
                variable_name="undl",
                value_name="ret_sq",
            )
            .with_columns(pl.col("undl").str.replace("ret_sq_", ""))
            .with_columns(
                day=pl.col("date").dt.day(),
                time=pl.col("date").dt.time(),
            )
            .sort(["date", "undl"])
        )
        self.df2 = df

        dfg = (
            df.group_by(["day", "undl"])
            .agg(
                time=pl.col("time"),
                ret_sq=pl.col("ret_sq"),
                ret_sq_cum_sum=pl.col("ret_sq").cum_sum(),
            )
            .sort(["day", "undl"])
        )
        self.df3 = dfg

        dfo = dfg.drop("ret_sq").explode(["time", "ret_sq_cum_sum"])
        self.df4 = dfo

        dfc = (
            dfo.group_by("day", "undl")
            .agg(
                n_tick=pl.col("ret_sq_cum_sum").len(),
                vol_rea=np.sqrt(
                    252
                    * n_tick_per_day
                    * (pl.col("ret_sq_cum_sum").max() / pl.col("ret_sq_cum_sum").len())
                ),
            )
            .sort(["undl", "day"])
        )

        self.df5 = dfc

    def check_volatility_precision(self):
        avg_vol_by_undl = (
            self.df5.group_by("undl")
            .agg(
                avg_realized_vol=pl.col("vol_rea").mean(),
            )
        )

        input_vols = pl.DataFrame(
            {
                "undl": ["A", "B"],
                "input_vol": [self.vol_A, self.vol_B],
            }
        )

        precision_metrics = (
            avg_vol_by_undl.join(input_vols, on="undl")
            .with_columns(
                absolute_error=pl.col("avg_realized_vol") - pl.col("input_vol"),
                relative_error=(
                    (pl.col("avg_realized_vol") - pl.col("input_vol")) / pl.col("input_vol")
                ),
                relative_error_pct=(
                    ((pl.col("avg_realized_vol") - pl.col("input_vol")) / pl.col("input_vol"))
                    * 100
                ),
            )
            .with_columns(
                absolute_error=pl.col("absolute_error").round(4),
                relative_error=pl.col("relative_error").round(4),
                relative_error_pct=pl.col("relative_error_pct").round(2),
            )
        )

        return precision_metrics


c = Trial()
c.build()

print(c.df1)
print(c.df2)
print(c.df3)
print(c.df4)
print(c.df5)
print(f"vol_A={c.vol_A}")
print(f"vol_B={c.vol_B}")
print(f"n_tick_per_day={c.n_tick_per_day}")

vol_precision = c.check_volatility_precision()
print("\nVolatility Precision Metrics:")
print(vol_precision)

daily_precision = (
    c.df5.with_columns(
        input_vol=pl.when(pl.col("undl") == "A")
        .then(c.vol_A)
        .otherwise(c.vol_B),
        error_pct=((pl.col("vol_rea") - pl.when(pl.col("undl") == "A")
                  .then(c.vol_A)
                  .otherwise(c.vol_B)) /
                  pl.when(pl.col("undl") == "A")
                  .then(c.vol_A)
                  .otherwise(c.vol_B)) * 100,
    )
    .select(["day", "undl", "vol_rea", "input_vol", "error_pct"])
    .with_columns(
        vol_rea=pl.col("vol_rea").round(4),
        error_pct=pl.col("error_pct").round(2)
    )
)
print("\nVolatility by Day vs Input:")
print(daily_precision)

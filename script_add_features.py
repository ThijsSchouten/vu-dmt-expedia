# Load data
import pandas as pd
from lib_data import *


def create_checkin_checkout(data):
    # data["check_in"] = data["date_time"].dt.dayofyear
    data["check_in"] = data["date_time"].apply(lambda x: x.timetuple().tm_yday)
    print(data["check_in"][0])
    data["check_in"] = data["check_in"] + data["srch_booking_window"]
    data["check_out"] = data["check_in"] + data["srch_length_of_stay"]
    return data


def create_price_ranks(data):
    # Takes dataset and adds the price_ranks
    # maybe TODO is add orginial price variable
    data["price_srch_id_rank"] = data.groupby(by="srch_id")[
        "price_srch_id"
    ].rank(method="dense", ascending=False)
    data["price_srch_destination_id_rank"] = data.groupby(by="srch_id")[
        "price_srch_destination_id"
    ].rank(method="dense", ascending=False)
    data["price_srch_booking_window_rank"] = data.groupby(by="srch_id")[
        "price_srch_booking_window"
    ].rank(method="dense", ascending=False)
    data["price_prop_country_id_rank"] = data.groupby(by="srch_id")[
        "price_prop_country_id"
    ].rank(method="dense", ascending=False)
    data["price_date_time_rank"] = data.groupby(by="srch_id")[
        "price_date_time"
    ].rank(method="dense", ascending=False)
    data["price_prop_id_rank"] = data.groupby(by="srch_id")[
        "price_prop_id"
    ].rank(method="dense", ascending=False)
    return data

from datetime import datetime, timedelta


def get_date(date_str, offset):
    return (datetime.strptime(date_str, "%m/%d/%y") + timedelta(days=offset)).strftime(
        "%-m/%-d/%y")

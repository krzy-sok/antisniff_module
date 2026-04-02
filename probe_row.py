from pydantic import BaseModel

class ProbeRow(BaseModel):
    rtt_avg: float
    rtt_median: float
    flood_flag: int
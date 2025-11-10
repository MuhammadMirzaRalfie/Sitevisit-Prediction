from pydantic import BaseModel

class SiteVisitInput(BaseModel):
    ORDERTYPE: str
    PRODFAMILYNAME: str
    FPA: str
    NEXT_PROCESS: str
    ISDS_ORDER_JAR: str
    MEDIAACCESS: str
    ORDERSUBTYPE: str
    SOURCESITELAT: float
    SOURCESITELONG: float
    PREVIOUSTASKNAME: str
    DESTINATIONPOPSITELAT: float
    DESTINATIONPOPSITELONG: float
    KAMSUBCATEGORY: str

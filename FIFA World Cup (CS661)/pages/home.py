from dash import html
import dash_bootstrap_components as dbc
# from components.WC_Header import WCHeaderCard
# from components.WCWinnerRegionSunburst import WCWinnerRegion
# from components.WCComponents import *
from components.introcard import WCIntroCard
from components.winners import WCHeaderCard
#from components.cloud import WCCloudCard
from components.venue_attend import VenuesAndCitiesBar, TotalAttendanceLine
from components.winner_region import WCWinnersBar, WCWinnerRegion, HostsCountriesBar
from components.timeline import WCToursTimeline, WCGoalspermin
from components.racebar import WCRacechart

worldcup_page_content = dbc.Container([
    dbc.Row([
            WCIntroCard,
            ]),
    dbc.Row([
        WCHeaderCard,
        dbc.Col([
        WCRacechart,
        VenuesAndCitiesBar,
        TotalAttendanceLine,
        WCWinnersBar,
        WCWinnerRegion,
        HostsCountriesBar,
        WCToursTimeline,
        WCGoalspermin,
        ], className="m-0 p-0")
    ]),
    dbc.Row([
        #VenuesAndCitiesBar,
        #TotalAttendanceLine,
            ]),
    dbc.Row([
        #WCWinnersBar,
        #WCWinnerRegion,
        #HostsCountriesBar,
    ]),
    dbc.Row([
            #WCToursTimeline,
            #WCGoalspermin,
            ]),
    dbc.Row([
            #WCRacechart,
            ]),
], style={"padding-top": "40px"})
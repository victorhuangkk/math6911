# This file should be under zipline's folder to ingest customized data

import pandas as pd
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities

start_session = pd.Timestamp('2018-1-1', tz='utc')
end_session = pd.Timestamp('2021-1-10', tz='utc')

register(
    'victor-csvdir-bundle',
    csvdir_equities(
        ['daily'],
        'C:/Users/16477/Desktop/zipline/dat',
    ),
    calendar_name='NYSE',
    start_session=start_session,
    end_session=end_session
)

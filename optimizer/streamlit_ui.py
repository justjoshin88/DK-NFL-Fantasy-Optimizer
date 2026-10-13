"""Contest → exact DK pool → automatic data retrieval → review → legal roster export."""
import hashlib
import io
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

if __package__:
    from .dk_csv import load_pool, read_text, export_rosters, roster_template
    from .providers import fetch_public, fetch_sportsdata, match_projections, fetch_contest, verify_contest_pool, load_projection_file, ProviderError
    from .lineup_engine import eligible_pool, build_portfolio
    from .calibration import calibrate, evaluate_portfolio
    from .free_model import fetch_free, load_datasets
else:
    from dk_csv import load_pool, read_text, export_rosters, roster_template
    from providers import fetch_public, fetch_sportsdata, match_projections, fetch_contest, verify_contest_pool, load_projection_file, ProviderError
    from lineup_engine import eligible_pool, build_portfolio
    from calibration import calibrate, evaluate_portfolio
    from free_model import fetch_free, load_datasets


def setting(name, default=''):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.environ.get(name, default)


@st.cache_data(ttl=300, show_spinner=False)
def automatic_data(pool, provider, key_fingerprint='', _key='', production=False):
    if provider == 'Free automatic':
        return fetch_free(pool, datasets=free_datasets(pool))
    if provider == 'SportsDataIO':
        if not _key:
            raise ProviderError('SPORTSDATAIO_API_KEY is not configured. Use public forecasts or configure a licensed real-data feed in Streamlit Secrets.')
        forecasts = fetch_sportsdata(pool, _key, production)
        return match_projections(pool, forecasts), []
    forecasts, messages = fetch_public(pool)
    return match_projections(pool, forecasts), messages


@st.cache_data(ttl=3600, show_spinner=False)
def free_datasets(pool):
    return load_datasets(pool)


@st.cache_data(ttl=300, show_spinner=False)
def contest_context(contest, pool):
    metadata = fetch_contest(contest)
    verify_contest_pool(metadata, pool)
    return metadata


def legacy_workflow(upload, strategy, optimizer_class):
    if __package__:
        from .data_loader import DataLoader
        from .strategy_manager import Strategy, StrategyManager
    else:
        from data_loader import DataLoader
        from strategy_manager import Strategy, StrategyManager
    st.info('Legacy custom projection file detected. Upload a native DraftKings pool to enable contest verification and DK roster export.')
    data = DataLoader().normalize_positions(DataLoader().load_data(io.StringIO(read_text(upload)))).reset_index(drop=True)
    st.dataframe(data, hide_index=True)
    if st.button('Build lineup', type='primary'):
        manager = StrategyManager(Strategy.GPP if strategy == 'GPP' else Strategy.CASH)
        optimizer = optimizer_class(data)
        lineup = optimizer.data.loc[optimizer.solve(*optimizer.build_model(manager.get_objective_column(data.columns), manager))]
        st.success('Lineup built successfully')
        st.dataframe(lineup, hide_index=True)
        st.download_button('Download lineup summary', lineup.to_csv(index=False), 'lineup_summary.csv', 'text/csv')


def main(optimizer_class=None):
    st.set_page_config(page_title='DK NFL Fantasy Optimizer', layout='wide')
    st.title('DK NFL Fantasy Optimizer')
    st.write('Choose your contest and NFL game type, upload its player pool, and retrieve matching pregame projections automatically.')
    st.caption('DraftKings IDs and salaries stay unchanged. Historical AvgPointsPerGame is never used as a forecast.')
    with st.sidebar:
        st.header('Contest setup')
        contest = st.text_input('DraftKings contest URL or ID', placeholder='https://www.draftkings.com/draft/contest/...')
        game_type = st.selectbox('Game type', ['Classic', 'Showdown'])
        strategy = st.selectbox('Strategy', ['Cash', 'GPP'])
        count = st.number_input('Lineups', 1, 20, 3)
        provider = st.selectbox('Projection source', ['Free automatic', 'Public forecasts', 'SportsDataIO', 'Upload projections'])
        auto = st.checkbox('Retrieve automatically after upload', value=True)
        with st.expander('Licensed feed setup'):
            st.write('For SportsDataIO, add SPORTSDATAIO_API_KEY and SPORTSDATAIO_PRODUCTION = true to Streamlit Secrets. Only enable production for a real-data subscription; trial data is scrambled.')
            st.link_button('SportsDataIO documentation', 'https://sportsdata.io/developers/api-documentation/nfl')
    upload = st.file_uploader('Player pool CSV', type=['csv'])
    template = st.file_uploader('Blank DraftKings roster template (optional)', type=['csv'])
    projections = st.file_uploader('Projection file (for Upload projections)', type=['csv']) if provider == 'Upload projections' else None
    with st.expander('Model controls and historical validation'):
        st.write('Cash maximizes forecast points. GPP uses QB stacks and lineup diversity. With a historical forecast archive, scenario candidates use empirical residuals; without it, no uncertainty or win rate is invented.')
        min_unique = st.number_input('Minimum different athletes between lineups', 1, 6 if game_type == 'Showdown' else 9, 2)
        max_exposure = st.slider('Maximum athlete exposure (%)', 10, 100, 100, 5)
        history = st.file_uploader('Historical pregame projections and final DK scores (optional)', type=['csv'])
        st.caption('History columns: game_date, position, projection, actual. Use at least eight game dates with saved pregame forecasts, not forecasts reconstructed after results.')
    with st.expander('File formats and model limits'):
        st.write('Use the contest-specific DraftKings pool with Position, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev. A roster-only template contains no player information.')
        st.write('Projection uploads need player_id and projection. Optional: ownership, injury_status, projection_ceiling, projection_std. Showdown projection values must be base FLEX scores for every row; the optimizer applies 1.5× for CPT.')
        st.write('No model guarantees a win. Forecast coverage, injury flags, ownership, and holdout diagnostics are shown for inspection. Payouts and opposing lineups are not automatically modeled; GPP recommendations are exploratory, not verified expected-profit estimates. Late-swap optimization is not supported.')
        st.write('Free automatic uses published exact-game forecasts where available and our own nflverse estimates elsewhere. The model learns only from completed past games and is tested on later dates. It requires active-roster identity and at least three observed games; rookies, changed roles and game-day injuries can remain uncovered. No API key or paid scraper is required.')
    if upload is None:
        st.info('Upload the exact contest player pool CSV to begin.')
        return
    digest = hashlib.sha256(read_text(upload).encode()).hexdigest()
    context = f'{digest}:{contest}:{game_type}:{strategy}:{provider}:{count}:{min_unique}:{max_exposure}'
    if st.session_state.get('context') != context:
        st.session_state['context'] = context
        st.session_state.pop('portfolio', None)
        st.session_state.pop('manual_data', None)
    try:
        lines = read_text(upload).splitlines()
        if not lines:
            raise ValueError('The player pool CSV is empty.')
        if 'player_name' in lines[0].split(','):
            legacy_workflow(upload, strategy, optimizer_class)
            return
        pool = load_pool(upload, game_type)
        roster_template(template, game_type)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(str(exc))
        return
    cols = st.columns(3)
    cols[0].metric('Pool rows', len(pool))
    cols[1].metric('Games', pool.game_id.nunique())
    cols[2].metric('Salary cap', '$50,000')
    st.caption('Game dates: ' + ', '.join(sorted(pool.game_date.unique())))
    if contest:
        try:
            with st.spinner('Checking the contest and player IDs...'):
                metadata = contest_context(contest, pool)
            st.success(f'Contest verified: {metadata["name"]} — all pool IDs belong to its draft group.')
            st.write({k: v for k, v in metadata.items() if k != 'raw' and v is not None})
        except ProviderError as exc:
            st.warning(f'Contest verification unavailable: {exc} The uploaded pool is used as supplied. Confirm it is the exact contest export.')
        except ValueError as exc:
            st.error(str(exc))
            return
    else:
        st.info('No contest ID entered. The uploaded CSV defines the slate; contest membership is unverified.')
    refresh = st.button('Refresh projections')
    if refresh:
        automatic_data.clear()
        st.session_state.pop('portfolio', None)
    try:
        if provider == 'Upload projections':
            if projections is None:
                st.info('Upload current pregame projections with DraftKings player IDs.')
                return
            data = load_projection_file(projections, pool)
            st.warning('Uploaded forecasts are user supplied. Their freshness and injury coverage cannot be independently verified.')
        elif auto or refresh:
            key = str(setting('SPORTSDATAIO_API_KEY'))
            production = str(setting('SPORTSDATAIO_PRODUCTION', False)).lower() == 'true'
            with st.spinner('Retrieving exact player and game forecasts...'):
                data, messages = automatic_data(pool, provider, hashlib.sha256(key.encode()).hexdigest(), key, production)
            st.session_state['manual_data'] = data
            for message in messages:
                st.warning(message)
        elif st.session_state.get('manual_data') is not None:
            data = st.session_state['manual_data']
        else:
            st.info('Enable automatic retrieval or click Refresh projections to load forecasts.')
            return
    except (ProviderError, ValueError, KeyError) as exc:
        st.error(str(exc))
        st.info('Future slate forecasts may not be published yet. Try again closer to kickoff, or use a licensed feed/current projection file. The app will not use another week or substitute season averages.')
        return
    # A new forecast/upload must invalidate older exports, even on a cache refresh.
    forecast_digest = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()
    if st.session_state.get('forecast_digest') != forecast_digest:
        st.session_state['forecast_digest'] = forecast_digest
        st.session_state.pop('portfolio', None)
    available, excluded = eligible_pool(data)
    if 'free_model_metrics' in data.attrs:
        st.warning(data.attrs['free_model_notes'])
        with st.expander('Free model: untouched chronological test results'):
            st.dataframe(data.attrs['free_model_metrics'], hide_index=True)
            st.caption('Older 60% of dates train the models, next 20% select ridge vs recency and regularization, newest 20% test the selection. Errors describe points, not win rates or profit. Models are then refitted on available past games for the current forecasts.')
        with st.expander('Free data attribution'):
            st.write('Historical statistics and roster data: nflverse contributors, CC BY 4.0. Schedule data: Lee Sharpe / nflverse nfldata. Data are transformed into our own scoring totals and forecasts.')
            st.link_button('nflverse data and license', 'https://github.com/nflverse/nflverse-data/blob/main/LICENSE.md')
            st.link_button('NFL schedules and source', 'https://github.com/nflverse/nfldata')
    st.subheader('Forecast coverage')
    cols = st.columns(3)
    cols[0].metric('Eligible forecast rows', len(available))
    cols[1].metric('Excluded rows', len(excluded))
    cols[2].metric('Ownership coverage', f'{available.ownership.notna().mean():.0%}' if 'ownership' in available and len(available) else 'Unavailable')
    if not excluded.empty:
        st.warning(f'{len(excluded)} pool rows are excluded for missing forecasts, injury status, or game lock. Optimization covers only the eligible subset.')
        with st.expander('Inspect excluded players'):
            st.dataframe(excluded[['player_name', 'position', 'team', 'game_date', 'exclusion_reason']], hide_index=True)
    if available.empty:
        st.error('No eligible current forecasts for this pool. No lineup can be generated.')
        return
    display = [c for c in ['player_name', 'player_id', 'position', 'team', 'opponent', 'game_date', 'salary', 'projection',
                           'injury_status', 'ownership', 'game_total', 'implied_team_total', 'source', 'model_type', 'history_games',
                           'last_played', 'training_last_game', 'source_updated', 'fetched_at'] if c in available]
    st.dataframe(available[display], hide_index=True)
    st.download_button('Download forecast audit', data.to_csv(index=False), 'forecast_audit.csv', 'text/csv')
    if 'source_url' in available:
        with st.expander('Sources'):
            for url in available.source_url.dropna().unique():
                st.write(url)
    calibration = None
    if history is not None:
        try:
            calibration, metrics = calibrate(pd.read_csv(io.StringIO(read_text(history))))
            if not set(available.position).issubset(calibration):
                raise ValueError('Historical calibration does not cover every eligible position.')
            st.dataframe(metrics, hide_index=True)
            st.caption('Holdout diagnostics describe historical forecast errors. Simulations assume independent position residuals; game correlations, ownership, and payouts are not calibrated.')
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
            return
    if strategy == 'GPP' and calibration is None:
        st.info('GPP uses supplied ceilings when complete; otherwise it optimizes mean forecasts with stacks and diversity. No assumed volatility or winning probability is shown.')
    names = dict(zip(available.player_key, available.player_name + ' (' + available.team + ')'))
    locks = st.multiselect('Lock athletes into every lineup', list(names), format_func=names.get)
    bans = st.multiselect('Exclude athletes', list(names), format_func=names.get)
    controls = (tuple(locks), tuple(bans), hashlib.sha256(read_text(history).encode()).hexdigest() if history is not None else '')
    if st.session_state.get('controls') != controls:
        st.session_state['controls'] = controls
        st.session_state.pop('portfolio', None)
    if st.button('Build lineups', type='primary'):
        try:
            with st.spinner('Solving salary, roster, game, and exposure constraints...'):
                lineups, message = build_portfolio(available, game_type, int(count), int(min_unique), max_exposure / 100,
                                                   strategy, locks, bans, calibration)
            st.session_state['portfolio'] = lineups
            st.session_state['portfolio_message'] = message
        except Exception as exc:
            st.error(f'Unable to build lineups: {exc}')
    lineups = st.session_state.get('portfolio')
    if lineups:
        # Do not offer stale rosters once any selected game has locked.
        if any((pd.to_datetime(lineup.kickoff, utc=True) <= datetime.now(timezone.utc)).any() for lineup in lineups):
            st.session_state.pop('portfolio', None)
            st.error('A selected game has started. These rosters are no longer available for new entries. Refresh the pool.')
            return
        st.success(f'{len(lineups)} legal lineups generated')
        if st.session_state.get('portfolio_message'):
            st.warning(st.session_state['portfolio_message'])
        for i, lineup in enumerate(lineups):
            st.subheader(f'Lineup {i + 1} · ${lineup.salary.sum():,} · {lineup.projected_points.sum():.2f} projected points')
            st.dataframe(lineup[['slot', 'player_name', 'player_id', 'team', 'salary', 'projected_points']], hide_index=True)
        exposure = pd.concat(lineups).groupby('player_key').agg(player=('player_name', 'first'), appearances=('player_name', 'count'))
        exposure['exposure_pct'] = exposure.appearances / len(lineups) * 100
        with st.expander('Portfolio exposure'):
            st.dataframe(exposure.reset_index(drop=True), hide_index=True)
        if calibration is not None:
            st.dataframe(evaluate_portfolio(lineups, calibration), hide_index=True)
        st.download_button('Download DraftKings roster CSV', export_rosters(lineups, game_type, template), 'DK_rosters.csv', 'text/csv')
        st.caption('ID-only roster upload in the exact position order. This is not an entry-edit CSV and does not submit or pay for entries.')

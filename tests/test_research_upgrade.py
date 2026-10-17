import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from optimizer.official_injuries import parse_official, apply_official, load_official
from optimizer.forecast_validation import fit_candidate, predict, prediction_interval
from optimizer.scenarios import sample_outcomes
from optimizer.providers import ProviderError


def page(status='Out', week=6, date='SUNDAY, OCTOBER 18TH'):
    def table(name):
        return '<table><thead><tr>'+''.join('<th>'+c+'</th>' for c in ['Player','Position','Injuries','Practice Status','Game Status'])+'</tr></thead><tbody><tr>'+''.join('<td>'+c+'</td>' for c in [name,'QB','Ankle','Limited Participation in Practice',status])+'</tr></tbody></table>'
    return f'''<meta property="og:url" content="https://www.nfl.com/injuries/league/2026/reg{week}">
    <h2 class="nfl-c-content-header__roofline">Injuries - WEEK {week}</h2>
    <div class="nfl-o-injury-report__wrap"><h2 class="d3-o-section-title">THURSDAY, OCTOBER 15TH</h2>
    <h2 class="d3-o-section-title">{date}</h2><section class="nfl-o-injury-report__unit">
    <span class="nfl-c-matchup-strip__team-abbreviation">AZ</span><span class="nfl-c-matchup-strip__team-abbreviation">LA</span>
    {table('Player A')}{table('Player B')}</section></div>'''


def schedule():
    return pd.DataFrame([dict(season=2026, week=6, game_type='REG', away_team='ARI', home_team='LAR',
                             gameday='2026-10-18', kickoff_time=pd.Timestamp('2026-10-18T20:00Z'))])


def pool():
    return pd.DataFrame([dict(player_name='Player A', player_key='a', player_id='1', position='QB',
                             team='ARI', opponent='LAR', game_date='2026-10-18', exclusion_reason='',
                             salary=5000, eligible_slots=['QB'], projection=20.)])


class ResearchUpgradeTests(unittest.TestCase):
    def test_official_reports_verify_week_teams_and_nearest_date_heading(self):
        reports = parse_official(page(), schedule(), 2026, 6, 'REG', '2026-10-17T12:00Z')
        self.assertEqual(set(reports.team), {'ARI','LAR'})
        self.assertEqual(len(reports), 2)
        result = apply_official(pool(), reports, pd.Timestamp('2026-10-17T12:02Z'))
        self.assertIn('official NFL', result.exclusion_reason.iloc[0])
        self.assertEqual(result.eligible_slots.iloc[0], ['QB'])
        self.assertEqual(result.projection.iloc[0], 20.)
        self.assertNotIn('source_updated', result)

    def test_wrong_week_or_schema_is_rejected(self):
        for html in [page(week=7),page().replace('Game Status','Different header')]:
            with self.assertRaises(ProviderError):
                parse_official(html, schedule(), 2026, 6, 'REG', '2026-10-17T12:00Z')

    def test_wrong_game_date_or_opponent_never_matches(self):
        with self.assertRaises(ProviderError):
            parse_official(page(date='SUNDAY, OCTOBER 11TH'), schedule(), 2026, 6, 'REG', '2026-10-17T12:00Z')
        reports = parse_official(page(), schedule(), 2026, 6, 'REG', '2026-10-17T12:00Z')
        data=pool();data['opponent']='EVIL'
        self.assertEqual(apply_official(data,reports,pd.Timestamp('2026-10-17T12:01Z')).official_status.iloc[0], 'Unknown')

    def test_stale_and_future_reports_are_not_used(self):
        reports=parse_official(page(),schedule(),2026,6,'REG','2026-10-17T12:00Z')
        for clock in ['2026-10-17T12:16Z','2026-10-17T11:59Z']:
            self.assertEqual(apply_official(pool(),reports,pd.Timestamp(clock)).exclusion_reason.iloc[0], '')

    def test_blank_report_does_not_clear_another_exclusion(self):
        reports=parse_official(page(status=''),schedule(),2026,6,'REG','2026-10-17T12:00Z')
        data=pool();data['exclusion_reason']='Roster inactive'
        result=apply_official(data,reports,pd.Timestamp('2026-10-17T12:01Z'))
        self.assertEqual(result.exclusion_reason.iloc[0], 'Roster inactive')
        self.assertEqual(result.official_status.iloc[0], 'Not designated')

    def test_conflicting_official_status_is_excluded(self):
        reports=parse_official(page(),schedule(),2026,6,'REG','2026-10-17T12:00Z')
        second=reports.copy();second['official_status']='QUESTIONABLE'
        result=apply_official(pool(),pd.concat([reports,second]),pd.Timestamp('2026-10-17T12:01Z'))
        self.assertIn('Conflicting',result.exclusion_reason.iloc[0])

    def test_official_provider_failure_is_optional_and_explained(self):
        with patch('optimizer.official_injuries.request',side_effect=ProviderError('Unavailable')):
            reports,urls,notes=load_official(schedule(),pd.Timestamp('2026-10-17T12:00Z'))
        self.assertTrue(reports.empty);self.assertTrue(urls);self.assertIn('Unavailable',notes[0])

    def test_ensemble_is_exact_weighted_prediction_and_reuses_fits(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(70,6));y=10+x[:,0]
        specs=[dict(method='ridge',regularization=100.,features=6),dict(method='boosted_trees',leaves=7,features=6)]
        cache={};a,b=[fit_candidate(s,x,y,cache) for s in specs]
        m=fit_candidate(dict(method='ensemble',components=specs,weights=[.25,.75],features=6),x,y,cache)
        self.assertIs(a,m['members'][0]);self.assertIs(b,m['members'][1])
        np.testing.assert_allclose(predict(m,x),.25*predict(a,x)+.75*predict(b,x))

    def test_absolute_interval_uses_correct_finite_sample_order_statistic(self):
        model=dict(residuals=list(range(10)),interval_method='absolute')
        # ceil((10+1)*.8)=9: ninth ordered absolute residual is 8.
        self.assertEqual(prediction_interval(model,2),(-6,10))
        with self.assertRaises(ValueError):prediction_interval(dict(residuals=[np.nan]),2)

    def test_turnover_downside_is_preserved_in_offensive_scenarios(self):
        players=pd.DataFrame([dict(player_id='1',player_key='q',position='QB',team='A',game_id='g',projection=1.)])
        outcomes=sample_outcomes(players,{'QB':{'residuals':[-5,-4,-3]}},100,np.random.default_rng(1))
        self.assertLess(outcomes['q'].max(),0)


if __name__=='__main__':unittest.main()

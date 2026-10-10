import unittest
from .resource_resume import pending_jobs


class Tests(unittest.TestCase):
    def test_only_explicit_user_interruption_can_restart_once(self):
        jobs = [dict(id=x) for x in ('a', 'b', 'c')]
        complete = dict(phase='online_a', attempt=0, status='COMPLETE')
        interrupted = dict(phase='online_b', attempt=0, status='INCOMPLETE', failure=dict(exit_code=-15))
        self.assertEqual(pending_jobs(jobs, [complete, interrupted], 'online_b'), [(jobs[1], 1), (jobs[2], 0)])
        self.assertEqual(pending_jobs([jobs[1]], [interrupted, dict(interrupted, attempt=1, status='COMPLETE')], 'online_b'), [])
        for rows, phase in [([dict(interrupted, failure=dict(exit_code=1))], 'online_b'),
                            ([interrupted], 'online_other'),
                            ([interrupted, dict(interrupted, attempt=1)], 'online_b')]:
            with self.assertRaises(ValueError):
                pending_jobs([jobs[1]], rows, phase)

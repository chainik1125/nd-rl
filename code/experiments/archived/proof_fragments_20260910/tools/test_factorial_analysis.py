"""Check shared-context cancellation and length balancing on a known factorial."""
import json
import numpy as np
from analyze_evaluations import linear_contrast, summarize, cluster_weights


def main():
    rows = [{'source_id': str(i), 'context_id': f'c{i % 8}',
             'length_bin': 'short' if i < 8 else 'long'} for i in range(24)]
    nuisance = np.linspace(-10, 10, 24)
    selection = np.array([1.] * 8 + [3.] * 16)
    reset = np.array([4.] * 8 + [2.] * 16)
    interaction = np.array([2.] * 8 + [6.] * 16)
    raw = {name: {'outcome': value} for name, value in {
        'uf': nuisance, 'sf': nuisance + selection,
        'ur': nuisance + reset, 'sr': nuisance + selection + reset + interaction}.items()}
    weights = cluster_weights(rows, 5000, 42)
    result = linear_contrast(raw, {'sr': 1, 'sf': -1, 'ur': -1, 'uf': 1})['outcome']
    np.testing.assert_allclose(result, interaction, atol=1e-14)
    summary, boot = summarize(result, rows, weights)
    # Equal length weighting yields (2+6)/2=4, not source-weighted 14/3.
    np.testing.assert_allclose(summary['balanced']['mean'], 4.)
    np.testing.assert_allclose(boot, 4.)
    # Correlated nuisance cancels before bootstrapping, yielding a point interval.
    np.testing.assert_allclose(summary['balanced']['context_bootstrap95'], [4., 4.])
    avg = linear_contrast(raw, {'sf': .5, 'sr': .5, 'uf': -.5, 'ur': -.5})['outcome']
    np.testing.assert_allclose(avg, selection + interaction / 2, atol=1e-14)
    for coefficients in ({}, {'missing': 1}, {'sf': 0}, {'sf': float('nan')}):
        try:
            linear_contrast(raw, coefficients)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid contrast: {coefficients}')
    print(json.dumps({'status': 'passed', 'checks': ['factorial interaction', 'equal length weighting',
                      'paired nuisance cancellation', 'marginal effect', 'invalid contrast rejection']}))


if __name__ == '__main__':
    main()

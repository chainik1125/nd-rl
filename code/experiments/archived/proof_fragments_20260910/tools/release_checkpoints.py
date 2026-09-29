"""Create a hash-bound evaluation release from a prospectively frozen recipe."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe')
    args = parser.parse_args()
    recipe_path = Path(args.recipe).resolve()
    recipe = json.loads(recipe_path.read_text())
    if recipe['role'] not in ('confirmation', 'final_confirmation'):
        raise ValueError('Only declared confirmation roles may be released')
    if recipe['role'] == 'final_confirmation' and not recipe.get('no_further_training'):
        raise ValueError('Final release requires training closure')
    for path, digest in recipe['prospective_input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Prospective input drift: ' + path)
    data_hash = sha(recipe['data'])
    population = json.loads(Path(recipe['population_manifest']).read_text())
    if population['population_sha256'][recipe['role']] != data_hash:
        raise ValueError('Population role/hash differs')
    checkpoints = {path: sha(path) for path in recipe['checkpoint_paths']}
    if len(checkpoints) != len(recipe['checkpoint_paths']):
        raise ValueError('Repeated checkpoint path')
    output = {'frozen': True, 'no_further_selector_changes': True,
              'no_further_training': recipe.get('no_further_training', False),
              'role': recipe['role'], 'data_sha256': [data_hash],
              'checkpoints': list(checkpoints.values()), 'checkpoint_paths': checkpoints,
              'recipe_sha256': sha(recipe_path), 'recipe': str(recipe_path),
              'prospective_input_sha256': recipe['prospective_input_sha256'],
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'decoding': recipe['decoding'],
              'payload_parsed_by_release': False}
    with Path(recipe['output']).open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'released_checkpoints': len(checkpoints), 'role': recipe['role'], 'output': recipe['output']}), flush=True)


if __name__ == '__main__':
    main()

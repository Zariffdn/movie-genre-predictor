import 'package:flutter/material.dart';

import 'about_page.dart';
import 'example_plots.dart';
import 'genre_model.dart';

/// Below this many known words, the results come with a warning.
const _fewKnownWords = 8;

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.model});

  final GenreModel model;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _plot = TextEditingController();
  Prediction? _prediction;
  var _nextExample = 0;

  @override
  void dispose() {
    _plot.dispose();
    super.dispose();
  }

  void _predict() {
    FocusScope.of(context).unfocus(); // hide the keyboard so the results are visible
    if (_plot.text.trim().isEmpty) {
      setState(() => _prediction = null);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Type or paste a plot summary first.')),
      );
      return;
    }
    setState(() => _prediction = widget.model.predict(_plot.text));
  }

  void _tryExample() {
    _plot.text = examplePlots[_nextExample % examplePlots.length];
    _nextExample++;
    _predict();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Movie Genre Predictor'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: 'About the model',
            onPressed: () => Navigator.of(context).push(MaterialPageRoute<void>(
              builder: (_) => AboutPage(metrics: widget.model.testMetrics),
            )),
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            TextField(
              controller: _plot,
              minLines: 5,
              maxLines: 12,
              textCapitalization: TextCapitalization.sentences,
              decoration: InputDecoration(
                labelText: 'Plot summary',
                hintText: 'Describe what happens in a movie. A paragraph works best.',
                border: const OutlineInputBorder(),
                alignLabelWithHint: true,
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear),
                  tooltip: 'Clear',
                  onPressed: () => setState(() {
                    _plot.clear();
                    _prediction = null;
                  }),
                ),
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton.icon(
                  onPressed: _predict,
                  icon: const Icon(Icons.auto_awesome),
                  label: const Text('Predict genres'),
                ),
                OutlinedButton.icon(
                  onPressed: _tryExample,
                  icon: const Icon(Icons.movie_outlined),
                  label: const Text('Try an example'),
                ),
              ],
            ),
            const SizedBox(height: 24),
            if (_prediction case final prediction?) _Results(prediction: prediction),
          ],
        ),
      ),
    );
  }
}

class _Results extends StatelessWidget {
  const _Results({required this.prediction});

  final Prediction prediction;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // With no known words, every genre's score is just its intercept, a fixed
    // number that says nothing about this text. (For Drama, sigmoid(intercept)
    // is 0.391, just above its 0.39 threshold, so the raw model would answer
    // "Drama" to anything.) Show no genres rather than a meaningless guess.
    if (prediction.knownWordCount == 0) {
      return Text(
        'None of these words are in the model\'s vocabulary, so it can\'t '
        'guess a genre. Try describing the plot in English.',
        style: theme.textTheme.bodyLarge,
      );
    }

    final predicted = prediction.predicted;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (prediction.knownWordCount < _fewKnownWords)
          Card(
            color: theme.colorScheme.errorContainer,
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Text(
                'The model knows only ${prediction.knownWordCount} '
                '${prediction.knownWordCount == 1 ? 'word' : 'words'} in this text, '
                'so these results are shaky. Try a longer description.',
                style: TextStyle(color: theme.colorScheme.onErrorContainer),
              ),
            ),
          ),
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8),
          child: Text(
            predicted.isEmpty ? 'No genre passed its threshold' : 'Predicted genres',
            style: theme.textTheme.titleLarge,
          ),
        ),
        for (final genre in predicted) _GenreCard(genre: genre),
        ExpansionTile(
          // A new key for each prediction resets initiallyExpanded.
          key: ObjectKey(prediction),
          title: const Text('Other genres'),
          initiallyExpanded: predicted.isEmpty,
          tilePadding: EdgeInsets.zero,
          children: [
            for (final genre in prediction.notPredicted)
              ListTile(
                dense: true,
                contentPadding: EdgeInsets.zero,
                title: Text(genre.genre),
                trailing: Text(
                    '${_percent(genre.probability, roundDown: true)}  (needs ${_percent(genre.threshold)})'),
              ),
          ],
        ),
      ],
    );
  }
}

class _GenreCard extends StatelessWidget {
  const _GenreCard({required this.genre});

  final GenrePrediction genre;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(child: Text(genre.genre, style: theme.textTheme.titleMedium)),
                Text(_percent(genre.probability), style: theme.textTheme.titleMedium),
              ],
            ),
            const SizedBox(height: 8),
            _ProbabilityBar(probability: genre.probability, threshold: genre.threshold),
            const SizedBox(height: 4),
            Text('Threshold ${_percent(genre.threshold)}', style: theme.textTheme.bodySmall),
            if (genre.topWords.isNotEmpty) ...[
              const SizedBox(height: 12),
              Text('Words that pointed here', style: theme.textTheme.labelMedium),
              const SizedBox(height: 4),
              Wrap(
                spacing: 6,
                runSpacing: 6,
                children: [
                  for (final word in genre.topWords)
                    Chip(label: Text(word.word), visualDensity: VisualDensity.compact),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// A probability bar with a tick mark at the genre's threshold.
///
/// Hidden from screen readers: the card already says the probability and the
/// threshold in text.
class _ProbabilityBar extends StatelessWidget {
  const _ProbabilityBar({required this.probability, required this.threshold});

  final double probability;
  final double threshold;

  @override
  Widget build(BuildContext context) {
    return ExcludeSemantics(
      child: LayoutBuilder(
        builder: (context, constraints) => SizedBox(
          height: 12,
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              Positioned.fill(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: LinearProgressIndicator(value: probability),
                ),
              ),
              Positioned(
                left: constraints.maxWidth * threshold - 1,
                top: -3,
                bottom: -3,
                child: Container(width: 2, color: Theme.of(context).colorScheme.onSurface),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A whole percentage. Use [roundDown] for a genre that missed its threshold.
/// Otherwise 61.7% against a 62% threshold would show "62% (needs 62%)",
/// which looks like it should have passed.
String _percent(double value, {bool roundDown = false}) =>
    '${roundDown ? (value * 100).floor() : (value * 100).round()}%';

import 'package:flutter/material.dart';

import 'genre_model.dart';

/// How the model works, how well it did on the test set, and dataset credits.
class AboutPage extends StatelessWidget {
  const AboutPage({super.key, required this.metrics});

  final TestMetrics metrics;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final (low, high) = metrics.macroF1Interval;
    return Scaffold(
      appBar: AppBar(title: const Text('About the model')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            const _Section(
              title: 'How it works',
              body: 'Everything runs on your phone, with no internet connection. '
                  'The app splits your text into words and weighs each one by how '
                  'unusual it is across movie plots (TF-IDF). Then 15 small logistic '
                  'regression models, one per genre, each turn those weights into a '
                  'probability. A genre is shown when its probability reaches a '
                  'threshold that was tuned for that genre. The words listed under a '
                  'genre are the ones that added the most to its score.',
            ),
            _Section(
              title: 'How good is it?',
              body: 'Scored on ${_withCommas(metrics.testMovies)} movies that were held back '
                  'during training and tuning. '
                  'Micro-F1 ${metrics.microF1.toStringAsFixed(2)}, '
                  'macro-F1 ${metrics.macroF1.toStringAsFixed(2)} '
                  '(95% interval ${low.toStringAsFixed(2)}-${high.toStringAsFixed(2)}). '
                  'F1 runs from 0 to 1. "Always yes" is the F1 you would get by '
                  'tagging every movie with that genre.',
            ),
            Table(
              // The "Always yes" column gets extra room so its header stays on one line.
              columnWidths: const {0: FlexColumnWidth(1.6), 1: FlexColumnWidth(0.8), 2: FlexColumnWidth(1.3)},
              defaultVerticalAlignment: TableCellVerticalAlignment.middle,
              children: [
                _row(theme, ['Genre', 'F1', 'Always yes', 'Movies'], header: true),
                for (final MapEntry(key: genre, value: m) in metrics.perGenre.entries)
                  _row(theme, [
                    genre,
                    m.f1.toStringAsFixed(2),
                    m.alwaysYesF1.toStringAsFixed(2),
                    _withCommas(m.support),
                  ]),
              ],
            ),
            const SizedBox(height: 16),
            const _Section(
              title: 'What it can\'t do',
              body: 'It learned from English Wikipedia plot summaries, so it works best '
                  'on a paragraph written in that style. The genre labels come from '
                  'Freebase and are incomplete, so some "wrong" answers are really '
                  'missing labels. It also picks up quirks of the data. For example, '
                  'the names of cartoon characters point toward Family, because many '
                  'Family-tagged movies are cartoon shorts.',
            ),
            const _Section(
              title: 'Credits',
              body: 'Trained on the CMU Movie Summary Corpus by David Bamman, '
                  'Brendan O\'Connor and Noah A. Smith, "Learning Latent Personas of '
                  'Film Characters", ACL 2013. The plot summaries come from Wikipedia. '
                  'The corpus is released under a Creative Commons Attribution-ShareAlike '
                  'licence, and the model file in this app is derived from it and shared '
                  'under the same licence.',
            ),
            const SelectableText('https://www.cs.cmu.edu/~ark/personas/'),
          ],
        ),
      ),
    );
  }

  TableRow _row(ThemeData theme, List<String> cells, {bool header = false}) {
    final style = header ? theme.textTheme.labelLarge : theme.textTheme.bodyMedium;
    return TableRow(children: [
      for (final (i, cell) in cells.indexed)
        Padding(
          // A gap before each number column, so "F1" and "Always yes" don't run together.
          padding: EdgeInsets.only(top: 4, bottom: 4, left: i == 0 ? 0 : 12),
          child: Text(cell, style: style, textAlign: i == 0 ? TextAlign.start : TextAlign.end),
        ),
    ]);
  }
}

/// 5760 -> "5,760". (The intl package could do this, but one helper isn't worth a dependency.)
String _withCommas(int n) => n.toString().replaceAllMapped(RegExp(r'\B(?=(\d{3})+$)'), (_) => ',');

class _Section extends StatelessWidget {
  const _Section({required this.title, required this.body});

  final String title;
  final String body;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: theme.textTheme.titleMedium),
          const SizedBox(height: 6),
          Text(body, style: theme.textTheme.bodyMedium),
        ],
      ),
    );
  }
}

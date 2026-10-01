import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:movie_genre_predictor/main.dart';
import 'package:movie_genre_predictor/src/example_plots.dart';
import 'package:movie_genre_predictor/src/genre_model.dart';

void main() {
  late GenreModel model;

  setUpAll(() {
    model = GenreModel.fromJson(
        jsonDecode(File('assets/model.json').readAsStringSync()) as Map<String, dynamic>);
  });

  testWidgets('shows predicted genres and their top words', (tester) async {
    await tester.pumpWidget(GenreApp(loadModel: () async => model));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), examplePlots.first); // the western
    await tester.tap(find.text('Predict genres'));
    await tester.pumpAndSettle();

    expect(find.text('Predicted genres'), findsOneWidget);
    final western = find.ancestor(of: find.text('Western'), matching: find.byType(Card));
    expect(western, findsOneWidget);
    expect(find.descendant(of: western, matching: find.text('cattle')), findsOneWidget);
  });

  testWidgets('warns when the text has too few known words', (tester) async {
    await tester.pumpWidget(GenreApp(loadModel: () async => model));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), 'A cowboy.');
    await tester.tap(find.text('Predict genres'));
    await tester.pumpAndSettle();

    // "A" is too short to be a token, so "cowboy" is the only known word.
    expect(find.textContaining('knows only 1 word in this text'), findsOneWidget);
  });

  testWidgets('an empty box asks for a plot instead of predicting', (tester) async {
    await tester.pumpWidget(GenreApp(loadModel: () async => model));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Predict genres'));
    await tester.pump(); // let the SnackBar appear

    expect(find.text('Type or paste a plot summary first.'), findsOneWidget);
    expect(find.text('Predicted genres'), findsNothing);
  });

  testWidgets('text with no known words shows no genres at all', (tester) async {
    // With no known words the raw model would still say "Drama" (its
    // intercept alone clears the threshold), so the app must not show it.
    await tester.pumpWidget(GenreApp(loadModel: () async => model));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), 'Zzyzx qwertyuiop blorfing!');
    await tester.tap(find.text('Predict genres'));
    await tester.pumpAndSettle();

    expect(find.textContaining('None of these words'), findsOneWidget);
    expect(find.text('Drama'), findsNothing);
    expect(find.text('Predicted genres'), findsNothing);
  });

  testWidgets('About page shows test scores and dataset credit', (tester) async {
    await tester.pumpWidget(GenreApp(loadModel: () async => model));
    await tester.pumpAndSettle();

    await tester.tap(find.byTooltip('About the model'));
    await tester.pumpAndSettle();

    expect(find.text('Western'), findsOneWidget); // a row of the scores table
    // The credits are further down; ListView only builds what's on screen.
    await tester.scrollUntilVisible(find.textContaining('CMU Movie Summary Corpus'), 200);
    expect(find.textContaining('CMU Movie Summary Corpus'), findsOneWidget);
  });
}

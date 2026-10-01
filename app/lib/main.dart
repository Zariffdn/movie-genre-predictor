import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'src/genre_model.dart';
import 'src/home_page.dart';

void main() => runApp(const GenreApp());

/// Reads the model bundled with the app. It never touches the network.
///
/// The JSON is 3.5 MB, so it is decoded on a background isolate with
/// compute(). That keeps the loading spinner moving instead of freezing the
/// UI for a moment. `cache: false` stops the bundle from keeping a second
/// copy of the 3.5 MB string in memory after it has been parsed.
Future<GenreModel> loadBundledModel() async {
  final text = await rootBundle.loadString('assets/model.json', cache: false);
  return compute(_parseModel, text);
}

GenreModel _parseModel(String text) =>
    GenreModel.fromJson(jsonDecode(text) as Map<String, dynamic>);

class GenreApp extends StatefulWidget {
  /// Tests pass their own [loadModel] so they don't depend on the asset bundle.
  const GenreApp({super.key, this.loadModel = loadBundledModel});

  final Future<GenreModel> Function() loadModel;

  @override
  State<GenreApp> createState() => _GenreAppState();
}

class _GenreAppState extends State<GenreApp> {
  // Created once. If it were created in build(), every rebuild would reload the model.
  late final Future<GenreModel> _model = widget.loadModel();

  @override
  Widget build(BuildContext context) {
    const seed = Colors.deepPurple;
    return MaterialApp(
      title: 'Movie Genre Predictor',
      theme: ThemeData(colorSchemeSeed: seed),
      darkTheme: ThemeData(colorSchemeSeed: seed, brightness: Brightness.dark),
      home: FutureBuilder<GenreModel>(
        future: _model,
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return Scaffold(
              body: Center(child: Text('Could not load the model:\n${snapshot.error}')),
            );
          }
          if (!snapshot.hasData) {
            return const Scaffold(body: Center(child: CircularProgressIndicator()));
          }
          return HomePage(model: snapshot.data!);
        },
      ),
    );
  }
}

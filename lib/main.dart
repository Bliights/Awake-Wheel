import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:audioplayers/audioplayers.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatelessWidget {
  const MyHomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          const DecoratedBox(
            decoration: BoxDecoration(
              image: DecorationImage(
                image: AssetImage('assets/backgroundViolet.jpg'),
                fit: BoxFit.cover,
              ),
            ),
          ),
          Center(
            child: Image.asset('assets/awake wheel page principale.png'),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              margin: const EdgeInsets.all(20),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.zero,
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30)
                    )
                ),
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => const CameraPage()),
                  );
                },
                child: Image.asset('assets/start accueil.png'),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  _CameraPageState createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  Timer? _timer;
  Timer? _alertTimer;
  bool _showRedBorder = false;

  @override
  void initState() {
    super.initState();
    _initializeControllerFuture = _initCamera();
    sendImageToServer();
    _timer = Timer.periodic(const Duration(seconds: 5), (timer) => sendImageToServer());
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first
    );
    _controller = CameraController(
      frontCamera,
      ResolutionPreset.max,
    );
    return _controller.initialize();
  }

  Future<void> sendImageToServer() async {
    if (!_controller.value.isInitialized) {
      print('Camera is not ready yet.');
      return;
    }

    try {

      final image = await _controller.takePicture();
      var uri = Uri.parse('http://10.1.160.175:5000/predict'); // Change to your server URL

      // Create a multipart request
      var request = http.MultipartRequest('POST', uri);

      // Attach the file
      request.files.add(
          http.MultipartFile(
              'file',
              File(image.path).readAsBytes().asStream(),
              File(image.path).lengthSync(),
              filename: image.path.split("/").last
          )
      );

      // Send the request
      var response = await request.send();


      if (response.statusCode == 200) {
        print('Image uploaded and prediction received.');
        final res = await http.Response.fromStream(response);
        final predictionResult = jsonDecode(res.body);
        print('Prediction: ${predictionResult['predicted_category']}');
        setState(() {
          if (predictionResult['predicted_category'] == 1 || predictionResult['predicted_category'] == 2) {
            _showRedBorder = true;
            startAlertTimer();
          }
        });

      } else {
        print('Failed to predict image: ${response.statusCode}');
      }
    } catch (e) {
      print('Error sending image: $e');
    }
  }

  AudioPlayer audioPlayer = AudioPlayer();
  void startAlertTimer() {
    // Jouer le son immédiatement si la condition est vraie
    if (_showRedBorder) {
      playSound();
    }

    // Continuer à jouer le son toutes les 6 secondes si la condition reste vraie
    _alertTimer = Timer.periodic(const Duration(seconds: 6), (timer) async {
      if (_showRedBorder) {
        playSound();
      } else {
        timer.cancel();
        await audioPlayer.stop();
      }
    });
  }

// Fonction pour jouer le son
  void playSound() async {
    await audioPlayer.setVolume(0.8);
    await audioPlayer.play(AssetSource('sounds/alert.mp3'));
  }

  @override
  void dispose() {
    _controller.dispose();
    _timer?.cancel();
    _alertTimer?.cancel();  // S'assurer de nettoyer le timer d'alerte
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          Container(color: Colors.black), // Black background for the camera page
          Align(
            alignment: Alignment.topLeft,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(10.0, 40.0, 10.0, 10.0),
              child: Image.asset('assets/awake wheel page principale.png', width: 100),
            ),
          ),
          Align(
            alignment: Alignment.topRight,
            child: Container(
              margin: const EdgeInsets.all(10.0),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.zero,
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30)
                    )
                ),
                onPressed: () {
                  Navigator.pop(context);
                  _showRedBorder = false;
                  _alertTimer?.cancel();
                  audioPlayer.stop();
                },
                child: Image.asset('assets/leave.png', width: 100),
              ),
            ),
          ),
          Align(
            alignment: Alignment.center,
            child: FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  if (snapshot.hasError) {
                    return Text('Erreur lors du chargement de la caméra: ${snapshot.error}');
                  }
                  return Container(
                    decoration: BoxDecoration(
                      border: _showRedBorder ? Border.all(color: Colors.red, width: 5) : null,
                    ),
                    child: CameraPreview(_controller),
                  );
                } else {
                  return const Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),
          if (_showRedBorder)  // Afficher le bouton "Stop" si _showRedBorder est vrai
            Center(
              child: ElevatedButton(
                onPressed: () {
                  setState(() {
                    _showRedBorder = false;
                    _alertTimer?.cancel();
                    audioPlayer.stop();
                  });
                },
                style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    textStyle: const TextStyle(fontSize: 24)
                ),
                child: const Text('STOP', style: TextStyle(fontSize: 24)),
              ),
            ),
        ],
      ),
    );
  }
}
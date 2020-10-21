import numpy as np
import tensorflow as tf
from preprocess import *

class RNN_Part1(tf.keras.Model):
	def __init__(self, vocab):
		"""
        The RNN_Part1 class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

		super(RNN_Part1, self).__init__()

		# TODO: initialize tf.keras.layers!
		# - tf.keras.layers.Embedding for embedding layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
		# - tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
		# - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN
		self.vocab = vocab
		vocab_size = len(vocab)
		output_dim_embedded = 32
		output_dim = 256
		self.embeddedLayer = tf.keras.layers.Embedding(vocab_size + 1, output_dim_embedded)
		self.lstm = tf.keras.layers.LSTM(output_dim, return_sequences=True) #change to a GRU trains faster than the LSTM
		self.dense1 = tf.keras.layers.Dense(output_dim, activation="relu")
		self.dense2= tf.keras.layers.Dense(vocab_size, activation="softmax")



	def call(self, inputs):
		"""
        - You must use an embedding layer as the first layer of your network 
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :return: the batch element probabilities as a tensor
        """
		# TODO: implement the forward pass calls on your tf.keras.layers!

		return self.dense2(self.dense1(self.lstm(self.embeddedLayer(inputs))))

	def loss(self, probs, labels):
		"""
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

		# We recommend using tf.keras.losses.sparse_categorical_crossentropy
		# https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

    	# TODO: implement the loss function with mask as described in the writeup
		mask = tf.less(labels, self.vocab[FIRST_SPECIAL]) | tf.greater(labels, self.vocab[LAST_SPECIAL])
		real_labels = tf.boolean_mask(labels, mask)
		real_probs = tf.boolean_mask(probs, mask)
		loss = tf.keras.losses.sparse_categorical_crossentropy(real_labels, real_probs)
		loss = tf.reduce_mean(loss)
		return loss

class RNN_Part2(tf.keras.Model):
	def __init__(self, french_vocab, english_vocab):

		super(RNN_Part2, self).__init__()

		french_vocab_size = len(french_vocab)
		english_vocab_size = len(english_vocab)

		self.french_vocab = french_vocab
		self.english_vocab = english_vocab

    	# TODO: initialize tf.keras.layers!
		output_dim_embedded = 32
		output_dim = 256
		self.embeddedFrenchLayer = tf.keras.layers.Embedding(french_vocab_size + 1, output_dim_embedded)
		self.embeddedEnglishLayer = tf.keras.layers.Embedding(english_vocab_size + 1, output_dim_embedded)

		self.dense1 = tf.keras.layers.Dense(output_dim, activation="relu")
		self.dense2 = tf.keras.layers.Dense(english_vocab_size, activation="softmax")

		self.RNNencoder = tf.keras.layers.GRU(output_dim, return_sequences=True, name="encoder", return_state=True)
		self.RNNdecoder = tf.keras.layers.GRU(output_dim, return_sequences=True, name="decoder", return_state=True)



	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
		# TODO: implement the forward pass calls on your tf.keras.layers!
		# Note 1: in the diagram there are two inputs to the decoder
		#  (the decoder_input and the hidden state of the encoder)
		#  Be careful because we don't actually need the predictive output
		#   of the encoder -- only its hidden state
		# Note 2: If you use an LSTM, the hidden_state will be the last two
		#   outputs of calling the rnn. If you use a GRU, it will just be the
		#   second output.


		#Encoding french an converting to english
		frenchEmbeddedOutput = self.embeddedFrenchLayer(encoder_input)
		englishEmbeddedOutput = self.embeddedEnglishLayer(decoder_input)
		_, hiddenEncoderOuptut = self.RNNencoder(frenchEmbeddedOutput)
		decodedOutput, _ = self.RNNdecoder(englishEmbeddedOutput, hiddenEncoderOuptut)
		return self.dense2(self.dense1(decodedOutput))

	def loss_function(self, probs, labels):
		"""
		Calculates the model cross-entropy loss after one forward pass.

		:param probs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""


		# When computing loss, we need to compare the output probs and labels with a shift
		#  of 1 to ensure a proper alignment. This is because we generated the output by passing
		#  in a *START* token and the encoded French state.
		#
		# - The labels should have the first token removed:
		#	 [*START* COSC440 is the best class. *STOP*] --> [COSC440 is the best class. *STOP*]
		# - The logits should have the last token in the window removed:
		#	 [COSC440 is the best class. *STOP* *PAD*] --> [COSC440 is the best class. *STOP*]

    	# TODO: implement the loss function with mask as described in the writeup
		english_mask = tf.less(labels, self.english_vocab[FIRST_SPECIAL]) | tf.greater(labels, self.english_vocab[LAST_SPECIAL])
		real_labels = tf.boolean_mask(labels, english_mask)[:-1]
		#remobe labels first token
		real_probs = tf.boolean_mask(probs, english_mask)[1:]
		#remove labels last token
		loss = tf.keras.losses.sparse_categorical_crossentropy(real_labels, real_probs)
		loss = tf.reduce_mean(loss)
		return loss

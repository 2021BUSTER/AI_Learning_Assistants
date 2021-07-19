class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()
    # batch normalization -> ReLu -> Conv Layer
    # 여기서 ReLu 같은 경우는 변수가 없는 Layer이므로 여기서 굳이 initialize 해주지 않는다. (call쪽에서 사용하면 되므로)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding="same")

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding="same")

    # identity를 어떻게 할지 정의
    # 원래 Residual Unit을 하려면 위의 순서로 진행한 뒤, 바로 X를 더해서 내보내면 되는데,
    # 이 X와 위의 과정을 통해 얻은 Feature map과 차원이 동일해야 더하기 연산이 가능할 것이므로
    # 즉, 위에서 filter_in과 filter_out이 같아야 한다는 의미이다.
    # 하지만, 다를 수 있으므로 아래와 같은 작업을 거친다.

        if filter_in == filter_out:
            self.identity = lambda x: x
        else:
            self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding="same")

  # 아래에서 batch normalization은 train할때와 inference할 때 사용하는 것이 달라지므로 옵션을 줄것이다.
    def call(self, x, training=False, mask=None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h


class ResnetLayer(tf.keras.Model):
    # 아래 arg 중 filter_in : 처음 입력되는 filter 개수를 의미
    # Resnet Layer는 Residual unit이 여러개가 있게끔해주는것이므로
    # filters : [32, 32, 32, 32]는 32에서 32로 Residual unit이 연결되는 형태
  def __init__(self, filter_in, filters, kernel_size):
    super(ResnetLayer, self).__init__()
    self.sequnce = list()
    # [16] + [32, 32, 32]
    # 아래는 list의 length가 더 작은 것을 기준으로 zip이 되어서 돌아가기 때문에
    # 앞의 list의 마지막 element 32는 무시된다.
    # zip([16, 32, 32, 32], [32, 32, 32])
    for f_in, f_out in zip([filter_in] + list(filters), filters):
      self.sequnce.append(ResidualUnit(f_in, f_out, kernel_size))

  def call(self, x, training=False, mask=None):
    for unit in self.sequnce:
      # 위의 batch normalization에서 training이 쓰였기에 여기서 넘겨 주어야 한다.
      x = unit(x, training=training)
    return x
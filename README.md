Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) TRT_NOEXCEPT {
  // output the result to channel
  // 为什么传入totalsize + 1, 1, 1还未知，求出每个检测的float数量，乘以最大数量，得到总size。
  // Detection 应该是结构体{float bbox[4], float conf, float class_id}
  // mMaxOutObject是前面提到的plugin成员变量。
  
  int total_size = mMaxOutObject * sizeof(Detection) / sizeof(float);
  // 打印调试信息
  std::cout << "[YoloLayerPlugin::getOutputDimensions] index=" << index
            << " nbInputDims=" << nbInputDims
            << " total_size=" << total_size
            << " -> return Dims3(" << (total_size + 1) << ", 1, 1)" 
            << std::endl;  
  return Dims3(total_size + 1, 1, 1);


class Dims3 : public Dims2
{
public:
    //!
    //! \brief Construct an empty Dims3 object.
    //!
    Dims3()
        : Dims3(0, 0, 0)
    {
    }

    //!
    //! \brief Construct a Dims3 from 3 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //! \param d2 The third element.
    //!
    Dims3(int32_t d0, int32_t d1, int32_t d2)
        : Dims2(d0, d1)
    {
        nbDims = 3;
        d[2] = d2;
    }
};

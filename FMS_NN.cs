namespace fms
{
    public class FMS_NN
    {
        //ニューラルネットワークの構成に関する変数
        public const int LAYOR_MAX = 3;   //層の最大数
        public const int IN_UNIT = 18;  //入力情報数
        public const int OUT_UNIT = 1;  //出力目標数
        public const int UNIT_MAX =  20;   //ユニットの最大数
        public const double ZI = 1.0;   //シグモイド関数での正規化に使用する定数
        public const double W_INIT = 0.1;   //初期重み作成に使用

        //教師データ作成用
        public const bool CORECT_MODO = false;//trueで教師データを取得，通常時はfalse
        public const int SIM_MODE = 1;//1ですべての問題を同一NNでスケジュールする，２で問題ごとにNNを変化させる
        public const int SELECT_RATE = 1;   //1だとすべてのデータを記録する

        //BPに関する変数
        public const int GRAD_MODE = 2;     //1:最急降下法（単純な勾配降下法），2:Adam
        public const int error_func = 1; //誤差関数（二乗誤差：１，交叉エントロピー：２）
        public const bool answer_correct_rate = false;//競合内正答率の表示
        public const double MAX_DATA = 10000;   //ファイル当たりの学習データ数を指定の数まで削減する
        public const int TEST_RATE = 0;   //テストデータの割合（基本的に3）
        public const int BP_TRIAL = 5;//初期重みを変化させて学習する回数
        public const int EPOCH = 100;
        public const int SPAN = 100;//全誤差出力周期
        public const int POPULATION_TYPE_BP = 1;//1:一様分布,2:正規分布,3:Heの初期値
        public const int BATCH = 500;//教師データの分割数（１以上でミニバッチを適用する:バッチ当たりのデータが多いとメモリ不足になる）
        public const double EPS = 0.1;//学習率
        public const double MOMENT = 0.95;//モーメンタム法で使用,未使用
        public const bool DROP_OUT = false;
        public const double RAMDA = 0;

        //教師データの統合
        public const int INTEGRATE_NUM = 1; //統合後のファイル数
        public const int EACH_FILE_TEA_NUM = 1000; //それぞれの教師問題数


        public class WEIGHT
        {
            private double weight;
            public WEIGHT right;
            public WEIGHT left;

            public WEIGHT()
            {
                weight = 0.0;
                right = null;
                left = null;
            }

            public void set_weight(double w) { weight = w; }
            public double get_weight() { return weight; }
        }

        public class UNIT
        {
            private int number; //層内でのユニット番号
            private int info_num; //入力情報番号
            private double input_value; //入力の総和
            private double output_value; //出力値
            private double delta;//BP用δ
            public WEIGHT weight;
            public UNIT right;
            public UNIT left;

            public UNIT()
            {
                number = 0;
                input_value = 0.0;
                output_value = 0.0;
                delta = 0.0;
                weight = null;
                right = null;
                left = null;
            }

            public void set_unit_num(int UnitNo) { number = UnitNo; }
            public int get_unit_num() { return number; }

            public void set_info_num(int InfoNo) { info_num = InfoNo; }
            public int get_info_num() { return info_num; }

            public void set_input_value(double ivalue) { input_value = ivalue; }
            public double get_input_value() { return input_value; }

            public void set_output_value(double ovalue) { output_value = ovalue; }
            public double get_output_value() { return output_value; }

            public void set_delta(double BP_delta) { delta = BP_delta; }
            public double get_delta() { return delta; }




            /****************************************************************************
            関数名：add_weight
            引数  ：w:重みデータ
            動作  ：重みデータの格納
            戻り値：なし
            ****************************************************************************/
            public void add_weight(double w)
            {
                WEIGHT new_ptr = new WEIGHT(), cur_ptr;

                cur_ptr = weight;

                new_ptr.right = null;
                new_ptr.set_weight(w);
                if (cur_ptr == null)
                {
                    weight = new_ptr;
                    new_ptr.left = null;
                }
                else
                {
                    while (!(cur_ptr.right == null))
                    {
                        cur_ptr = cur_ptr.right;
                    }
                    cur_ptr.right = new_ptr;
                    new_ptr.left = cur_ptr;
                }
            }
        }



        public class UNIT_LIST
        {
            private int layor_num; //層の番号
            private int unit_max; //ユニットの数
            public UNIT unit;
            public UNIT_LIST right;
            public UNIT_LIST left;

            public UNIT_LIST()
            {
                layor_num = 0;
                unit_max = 0;
                unit = null;
                right = null;
                left = null;
            }

            public void set_layor_num(int LayorNo) { layor_num = LayorNo; }
            public int get_layor_num() { return layor_num; }

            public void set_unit_max(int UnitMax) { unit_max = UnitMax; }
            public int get_unit_max() { return unit_max; }


            /****************************************************************************
  　　　　　関数名：add_unit
  　　　　　引数  ：UnitNo:ユニット番号,InfoNo:入力情報番号
  　　　　　動作  ：ユニットの作成  ユニット番号，入力情報番号の格納
  　　　　　戻り値：なし
　　　　　　****************************************************************************/
            public void add_unit(int UnitNo, int InfoNo)
            {
                UNIT new_ptr = new UNIT(), cur_ptr;

                cur_ptr = unit;

                new_ptr.right = null;
                new_ptr.set_unit_num(UnitNo);
                new_ptr.set_info_num(InfoNo);
                if (cur_ptr == null)
                {
                    unit = new_ptr;
                    new_ptr.left = null;
                }
                else
                {
                    while (!(cur_ptr.right == null))
                    {
                        cur_ptr = cur_ptr.right;
                    }
                    cur_ptr.right = new_ptr;
                    new_ptr.left = cur_ptr;
                }
            }
        }
    }
}
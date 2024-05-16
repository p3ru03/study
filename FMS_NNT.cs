using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Windows.Forms;
using System.Threading.Tasks;

namespace fms
{
    public class FMS_NNT : LAN
    {
        private int TrainingMethod = 0; //学習方法の分岐に使用
        private FMS_FILE file;  //ファイルの名前読み込み用
        public Random wrand;
        public RandomBoxMuller random_box_muller;

        // ガントチャート用
        public static int nowTrial;          //現在のトライアル数
        public static int nowGeneration;    //現在の世代数

        private FMS_SIM simulator;

        private double[,,] BP_weight_data = new double[FMS_NN.LAYOR_MAX + 1, FMS_NN.UNIT_MAX + 1, FMS_NN.UNIT_MAX + 1];

        //ドロップアウト使用時に使用：0で残す，1で隠すユニット
        private double[,] drop_unit = new double[FMS_NN.LAYOR_MAX + 1, FMS_NN.UNIT_MAX + 1];

        //全教師データ管理
        private List<TEA_SAMPLE> list = new List<TEA_SAMPLE>();
        //全教師データ管理
        private List<TEA_SAMPLE_MAIN> main_list = new List<TEA_SAMPLE_MAIN>();
        //全教師データ管理
        private List<TEA_SAMPLE_MAIN> main_min_list = new List<TEA_SAMPLE_MAIN>();
        //テストデータ管理
        private List<TEA_SAMPLE_MAIN> test_list = new List<TEA_SAMPLE_MAIN>();
        //バッチ内データ管理
        private List<TEA_SAMPLE> min_list = new List<TEA_SAMPLE>();
        //出力目標値成型時データを記録
        private double[,] min_max = new double[2, FMS_NN.OUT_UNIT];
        //バッチ内平均勾配保存用
        private double[,,] ave_grad = new double[FMS_NN.LAYOR_MAX, FMS_NN.UNIT_MAX + 1, FMS_NN.UNIT_MAX + 1];
        //１ステップ前の更新サイズの保管（モーメンタム法用）
        private double[,,] moment = new double[FMS_NN.LAYOR_MAX, FMS_NN.UNIT_MAX + 1, FMS_NN.UNIT_MAX + 1];
        //１ステップ前の更新サイズの保管（RMSprop法用）
        private double[,,] rms = new double[FMS_NN.LAYOR_MAX, FMS_NN.UNIT_MAX + 1, FMS_NN.UNIT_MAX + 1];
        //各競合数毎の正答率のカウント
        private double[,,] rate_by_conf = new double[2, 2, FMS_BASE.JOB_MAX + 1];//[訓練1・テスト2，正解数1・総競合数2,競合数]
        //教師データ統合用配列
        private double[,,] inte_result = new double[FMS_NN.INTEGRATE_NUM, FMS_NN.EACH_FILE_TEA_NUM, 2];
        //教師データ統合用リスト
        private List<TEA_SAMPLE_MAIN> integrate_list = new List<TEA_SAMPLE_MAIN>();
        private List<TEA_SAMPLE_MAIN> integrate_sub_list = new List<TEA_SAMPLE_MAIN>();


        public FMS_NNT(GanttChartForm form)
        {
            this.form = form;

            //使うのはNN
            FMS_BASE.GT_MIXRULE = 1000;
            FMS_BASE.GA_MIX_RULE = FMS_BASE.GT_MIXRULE;
            FMS_BASE.GA_MIX_RULE2 = 1000;
            FMS_BASE.isTest = true;

            switch (MainForm.NNT_METHOD)
            {
                case 1:
                    TrainingMethod = 1;
                    break;

                case 2:
                    TrainingMethod = 2;
                    break;

                default:
                    TrainingMethod = 0;
                    break;

            }
        }

        public void nnt_run()
        {
            switch (TrainingMethod)
            {
                case 1:
                    if (FMS_BASE.GA_MIX_RULE2 == 1000)
                    {
                        BP_learn_for_NN2();
                    }

                    else
                    {
                        BP_learn();
                    }

                    break;

                case 2:
                    integrate_TEA();
                    break;

                default:
                    break;
            }
        }

        /******************************************************************************
       関数名：integrate_TEA()
       引  数：なし
       動  作：BPを利用した勾配降下法によるNNの学習
       戻り値：なし
       ******************************************************************************/
        public void integrate_TEA()
        {
            file = new FMS_FILE();


            //教師データ統合
            integrate_data();
        }


        /******************************************************************************
        関数名：BP_learn()
        引  数：なし
        動  作：BPを利用した勾配降下法によるNNの学習
        戻り値：なし
        ******************************************************************************/
        public void BP_learn()
        {
            int i, j, k, time, epoch, trial, conf_num, correct1, correct2;
            int true_conf1 = 0;
            int true_conf2 = 0;
            int min_batch, batch_num;
            double eps, gosa, best_gosa, best_result, abs_weight = 0, e, max;
            double[] result = new double[4];
            double[] tr_gosa = new double[FMS_NN.OUT_UNIT];
            double[] test_gosa = new double[FMS_NN.OUT_UNIT];
            simulator = new FMS_SIM(form);
            e = Math.Pow(10, -30);

            //ガントチャート表示なし
            simulator.doDrawGantt = false;

            host_computer = new HOST_COMPUTER(form);
            file = new FMS_FILE();

            string WeightFile = file.FILENAME_NN_WEIGHTS_UPDATE; //フォルダ名"\\nn\\nnt\\weights\\"

            //ファイルを削除し新規作成
            if (File.Exists(WeightFile) == false)
                Directory.Delete(WeightFile, true);
            Directory.CreateDirectory(WeightFile);
            Directory.CreateDirectory(WeightFile + "weight");

            record_file_data = true;


            //教師データの読み込み
            read_tea_date();

            //データの正規化
            data_normalize();

            //データの白色化
            //data_white();

            //データの変形:使わない
            //forming_data();

            //教師データからテスト用データを抜き取る:修正しないと使えない
            //remove_data();

            //初期値を変えて学習
            for (trial = 0; trial < FMS_NN.BP_TRIAL; trial++)
            {
                time = 1;
                //大きい数を設定する．
                result[0] = 100000;
                best_result = 1000000;
                best_gosa = 1000000;

                Directory.Delete(WeightFile + "weight", true);
                Directory.CreateDirectory(WeightFile + "weight");
                Directory.CreateDirectory(WeightFile + "トライアル" + (trial + 1) + "回目");

                StreamWriter fout;
                //NN構成を読み取る
                read_nn_con();
                //初期重みの作成
                CreateNNWeight(trial);

                //勾配降下法で重みを更新する回数
                for (epoch = 0; epoch < FMS_NN.EPOCH; epoch++)
                {
                    //不要？
                    eps = FMS_NN.EPS;
                    //バッチノーマライゼーションにおける平均分散を削除
                    for (i = 1; i < host_computer.get_layor_max(); i++)
                    {
                        for (j = 0; j < host_computer.get_unit_num(i); j++)
                        {
                            dis_b[i, j] = 1;
                            ave_b[i, j] = 0;
                        }
                    }

                    wrand = new Random(epoch);

                    //ミニバッチ内の教師事例を入れ替える
                    List<int> numbers = new List<int>();
                    for (i = 0; i <= main_list.Count; i++)
                    {
                        numbers.Add(i);
                    }
                    main_list.ForEach(d =>
                    {
                        int index = wrand.Next(0, numbers.Count);

                        d.batch = numbers[index] % FMS_NN.BATCH;

                        numbers.RemoveAt(index);

                    });
                    numbers.Clear();



                    //ミニバッチ方式によるBP法を行うため，バッチ数分繰り返す
                    for (min_batch = 0; min_batch < FMS_NN.BATCH; min_batch++)
                    {
                        //ミニバッチの作成
                        make_min_batch(min_batch);

                        //ミニバッチ内要素数のカウント
                        batch_num = min_list.Count(obj => obj.batch == min_batch);

                        //ドロップアウトユニットの選定
                        drop_out();

                        //バッチ内にデータがあるときのみ以下を行う
                        if (batch_num != 0)
                        {
                            Console.Write(".");
                            for (int layor = 1; layor < FMS_NN.LAYOR_MAX; layor++)
                            {
                                if (layor != 1)
                                    batch_normalize(layor - 1);

                                Parallel.For(0, batch_num, iii =>
                                {
                                    //バッチデータ内第i問における入力情報に対する出力値の算出
                                    nn_run(iii, layor);

                                });
                            }
                            Parallel.For(0, batch_num, iii =>
                            {
                                //バッチデータ内第iii問におけるδの算出
                                cal_delta(iii);

                                //第i問における勾配を算出
                                cal_gradient(iii);
                            });

                            //バッチ内平均勾配
                            cal_batch_gradient_2();

                            //NNの重みの更新
                            switch (FMS_NN.GRAD_MODE)
                            {
                                case 1:
                                    update_weight(eps, time);
                                    break;

                                case 2:
                                    update_weight_2(eps, time);
                                    break;
                            }

                        }
                        //データをmain_listに返還
                        return_batch_data();

                    }
                    //adam用
                    time++;

                    Console.WriteLine("現在の更新回数は" + epoch + "回です．");
                    switch (FMS_NN.error_func)
                    {
                        case 1://二乗誤差
                            //訓練誤差
                            for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                                tr_gosa[i] = main_list.AsParallel().Average(obj => Math.Pow((obj.output[i] - obj.target[i]), 2));

                            gosa = 0;
                            //テスト誤差
                            if (FMS_NN.TEST_RATE > 0)
                            {
                                convert_list();
                                Parallel.For(0, min_list.Count(), iii => nn_run_2(iii));
                                for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                                {
                                    test_gosa[i] = min_list.AsParallel().Average(obj => Math.Pow((obj.output[FMS_NN.LAYOR_MAX - 1, i] - obj.target[i]), 2));
                                    gosa += test_gosa[i];
                                }
                            }

                            break;

                        case 2://交差エントロピー
                            //訓練誤差
                            for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                            {
                                tr_gosa[i] = -main_list.AsParallel().Average(obj =>
                                    (obj.target[i] * Math.Log(obj.output[i] + e))
                                    + ((1 - obj.target[i]) * Math.Log(1 - obj.output[i] + e))
                                );
                            }
                            gosa = 0;
                            //テスト誤差
                            if (FMS_NN.TEST_RATE > 0)
                            {
                                convert_list();
                                Parallel.For(0, min_list.Count(), iii => nn_run_2(iii));
                                for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                                {
                                    test_gosa[i] = -min_list.AsParallel().Average(obj =>
                                        obj.target[i] * Math.Log(obj.output[FMS_NN.LAYOR_MAX - 1, i])
                                    + (1 - obj.target[i]) * Math.Log(1 - obj.output[FMS_NN.LAYOR_MAX - 1, i])
                                    );
                                    gosa += test_gosa[i];
                                }
                            }
                            break;


                    }


                    //重みの総和算出用
                    //abs_weight = cal_nn_weight_abs_ave();

                    //教師事例と出力の誤差の平均
                    for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                    {
                        Console.WriteLine("訓練誤差" + i + "は" + tr_gosa[i] + "です．");
                        Console.WriteLine("テスト誤差" + i + "は" + test_gosa[i] + "です．");
                    }

                    //バッチノーマライゼーションにおける分散を計算
                    for (i = 1; i < host_computer.get_layor_max(); i++)
                    {
                        for (j = 0; j < host_computer.get_unit_num(i); j++)
                        {
                            dis_b[i, j] = Math.Sqrt(dis_b[i, j] / main_list.Count());
                            ave_b[i, j] = ave_b[i, j] / FMS_NN.BATCH;
                        }
                    }

                    //優先度のばらつきを観察する用
                    string path;

                    path = Application.StartupPath;
                    if (!path.EndsWith("\\"))
                    {
                        path += "\\";
                    }
                    path = "priority";
                    string result_file = path + "\\result.dat";
                    File.Delete(result_file);


                    //学習の進行状況を確認するためシミュレーション
                    simulator.read_nn_con();
                    int selection_type = simulator.select_selection_type2();

                    //重みの格納
                    for (i = 1; i < FMS_NN.LAYOR_MAX; i++)
                    {

                        for (j = 0; j < FMS_NN.UNIT_MAX + 1; j++)
                        {
                            for (k = 0; k < FMS_NN.UNIT_MAX + 1; k++)
                            {
                                w[i, j, k] = BP_weight_data[i, j, k];
                            }
                        }
                    }
                    if (epoch == 0)
                        all_input_rec = true;


                    simulator.nnt_run(selection_type);

                    //適用対象の問題に対して正規化，白色化を行う
                    if (epoch == 0)
                    {
                        data_normalize_for_apply();
                        //data_white_for_apply();
                        all_input_rec = false;
                    }

                    if (FMS_BASE.FILE_MAX == 1)
                    {
                        result[0] = simulator.get_performance_func(0);
                        Console.WriteLine(result);
                    }
                    else
                    {
                        result = simulator.host_computer.result_statistics_for_nn();
                    }

                    //最小値が更新される場合のみ重みを保存
                    if (epoch == 0 || result[0] < best_result)
                    {
                        save_BP_weight();
                        best_result = result[0];
                    }

                    //評価値を保存
                    string update_result_file = file.FILENAME_NN_BP_UPDATE_RESULT;
                    update_result_file = update_result_file + ".dat";
                    fout = new StreamWriter(update_result_file, true, Encoding.GetEncoding("SHIFT_JIS"));
                    for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                    {
                        fout.Write(tr_gosa[i] + "\t");
                        fout.Write(test_gosa[i] + "\t");
                    }
                    fout.Write(abs_weight + "\t");
                    if (FMS_NN.answer_correct_rate == true)
                    {
                        fout.Write(correct1 + "\t");
                        fout.Write(true_conf1 + "\t");
                        fout.Write(correct2 + "\t");
                        fout.Write(true_conf2 + "\t");
                    }
                    fout.Write(result[0] + "\t");
                    fout.Write(result[1] + "\t");
                    fout.Write(result[2] + "\t");
                    fout.WriteLine(result[3]);

                    fout.Close();

                    if (epoch == 0 || epoch % FMS_NN.SPAN == 0)
                        //すべてのデータの誤差を出力する
                        write_out_noize(update_result_file);


                    Console.WriteLine();
                }//end of epoch


                //新しくファイルを作成しコピー
                DirectoryInfo dir = new DirectoryInfo(WeightFile + "weight");
                DirectoryInfo[] dirs = dir.GetDirectories();
                FileInfo[] files = dir.GetFiles();
                foreach (FileInfo file in files)
                {
                    string temppath = Path.Combine(WeightFile + "トライアル" + (trial + 1) + "回目", file.Name);
                    file.CopyTo(temppath, false);
                }

            }//end of trial


        }


        /******************************************************************************
        関数名：BP_learn()
        引  数：なし
        動  作：BPを利用した勾配降下法によるNNの学習
        戻り値：なし
        ******************************************************************************/
        public void BP_learn_for_NN2()
        {
            int i, j, k, time, epoch, trial, conf_num, correct1, correct2;
            int true_conf1 = 0;
            int true_conf2 = 0;
            int min_batch, batch_num;
            double eps, gosa, best_gosa, best_result, abs_weight = 0, e, max;
            double[] result = new double[4];
            double[] tr_gosa = new double[FMS_NN.OUT_UNIT];
            double[] test_gosa = new double[FMS_NN.OUT_UNIT];
            simulator = new FMS_SIM(form);
            e = Math.Pow(10, -30);

            //ガントチャート表示なし
            simulator.doDrawGantt = false;

            host_computer = new HOST_COMPUTER(form);
            file = new FMS_FILE();

            string WeightFile = file.FILENAME_NN_WEIGHTS_UPDATE; //フォルダ名"\\nn\\nnt\\weights\\"

            //ファイルを削除し新規作成
            if (File.Exists(WeightFile) == false)
                Directory.Delete(WeightFile, true);
            Directory.CreateDirectory(WeightFile);
            Directory.CreateDirectory(WeightFile + "weight");

            record_file_data = true;


            //教師データの読み込み
            read_tea_date();

            //NN構成を読み取る
            read_nn_con();

            if (FMS_BASE.GA_MIX_RULE == 1000)
            {
                //ジョブ選択場面で使用する重みの読み込み
                read_nn_w();
            }

            //データの正規化
            data_normalize2();

            //データの白色化
            //data_white();

            //データの変形:使わない
            //forming_data();

            //教師データからテスト用データを抜き取る:修正しないと使えない
            //remove_data();

            //初期値を変えて学習
            for (trial = 0; trial < FMS_NN.BP_TRIAL; trial++)
            {
                time = 1;
                //大きい数を設定する．
                result[0] = 100000;
                best_result = 1000000;
                best_gosa = 1000000;

                Directory.Delete(WeightFile + "weight", true);
                Directory.CreateDirectory(WeightFile + "weight");
                Directory.CreateDirectory(WeightFile + "トライアル" + (trial + 1) + "回目");

                StreamWriter fout;

                //初期重みの作成
                CreateNNWeight(trial);

                //勾配降下法で重みを更新する回数
                for (epoch = 0; epoch < FMS_NN.EPOCH; epoch++)
                {
                    //不要？
                    eps = FMS_NN.EPS;
                    //バッチノーマライゼーションにおける平均分散を削除
                    for (i = 1; i < host_computer.get_layor_max(); i++)
                    {
                        for (j = 0; j < host_computer.get_unit_num(i); j++)
                        {
                            dis_b[i, j] = 1;
                            ave_b[i, j] = 0;
                        }
                    }

                    wrand = new Random(epoch);

                    //ミニバッチ内の教師事例を入れ替える
                    List<int> numbers = new List<int>();
                    for (i = 0; i <= main_list.Count; i++)
                    {
                        numbers.Add(i);
                    }
                    main_list.ForEach(d =>
                    {
                        int index = wrand.Next(0, numbers.Count);

                        d.batch = numbers[index] % FMS_NN.BATCH;

                        numbers.RemoveAt(index);

                    });
                    numbers.Clear();



                    //ミニバッチ方式によるBP法を行うため，バッチ数分繰り返す
                    for (min_batch = 0; min_batch < FMS_NN.BATCH; min_batch++)
                    {
                        //ミニバッチの作成
                        make_min_batch(min_batch);

                        //ミニバッチ内要素数のカウント
                        batch_num = min_list.Count(obj => obj.batch == min_batch);

                        //ドロップアウトユニットの選定
                        drop_out();

                        //バッチ内にデータがあるときのみ以下を行う
                        if (batch_num != 0)
                        {
                            Console.Write(".");
                            for (int layor = 1; layor < FMS_NN.LAYOR_MAX; layor++)
                            {
                                if (layor != 1)
                                    batch_normalize(layor - 1);

                                Parallel.For(0, batch_num, iii =>
                                {
                                    //バッチデータ内第i問における入力情報に対する出力値の算出
                                    nn_run(iii, layor);

                                });
                            }
                            Parallel.For(0, batch_num, iii =>
                            {
                                //バッチデータ内第iii問におけるδの算出
                                cal_delta(iii);

                                //第i問における勾配を算出
                                cal_gradient(iii);
                            });

                            //バッチ内平均勾配
                            cal_batch_gradient_2();

                            //NNの重みの更新
                            switch (FMS_NN.GRAD_MODE)
                            {
                                case 1:
                                    update_weight(eps, time);
                                    break;

                                case 2:
                                    update_weight_2(eps, time);
                                    break;
                            }

                        }
                        //データをmain_listに返還
                        return_batch_data();

                    }
                    //adam用
                    time++;

                    Console.WriteLine("現在の更新回数は" + epoch + "回です．");
                    switch (FMS_NN.error_func)
                    {
                        case 1://二乗誤差
                            //訓練誤差
                            for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                                tr_gosa[i] = main_list.AsParallel().Average(obj => Math.Pow((obj.output[i] - obj.target[i]), 2));

                            gosa = 0;
                            //テスト誤差
                            if (FMS_NN.TEST_RATE > 0)
                            {
                                convert_list();
                                Parallel.For(0, min_list.Count(), iii => nn_run_2(iii));
                                for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                                {
                                    test_gosa[i] = min_list.AsParallel().Average(obj => Math.Pow((obj.output[FMS_NN.LAYOR_MAX - 1, i] - obj.target[i]), 2));
                                    gosa += test_gosa[i];
                                }
                            }

                            break;

                        case 2://交差エントロピー
                            //訓練誤差
                            for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                            {
                                tr_gosa[i] = -main_list.AsParallel().Average(obj =>
                                    (obj.target[i] * Math.Log(obj.output[i] + e))
                                    + ((1 - obj.target[i]) * Math.Log(1 - obj.output[i] + e))
                                );
                            }
                            gosa = 0;
                            //テスト誤差
                            if (FMS_NN.TEST_RATE > 0)
                            {
                                convert_list();
                                Parallel.For(0, min_list.Count(), iii => nn_run_2(iii));
                                for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                                {
                                    test_gosa[i] = -min_list.AsParallel().Average(obj =>
                                        obj.target[i] * Math.Log(obj.output[FMS_NN.LAYOR_MAX - 1, i])
                                    + (1 - obj.target[i]) * Math.Log(1 - obj.output[FMS_NN.LAYOR_MAX - 1, i])
                                    );
                                    gosa += test_gosa[i];
                                }
                            }
                            break;


                    }


                    //重みの総和算出用
                    //abs_weight = cal_nn_weight_abs_ave();

                    //教師事例と出力の誤差の平均
                    for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                    {
                        Console.WriteLine("訓練誤差" + i + "は" + tr_gosa[i] + "です．");
                        Console.WriteLine("テスト誤差" + i + "は" + test_gosa[i] + "です．");
                    }

                    //バッチノーマライゼーションにおける分散を計算
                    for (i = 1; i < host_computer.get_layor_max(); i++)
                    {
                        for (j = 0; j < host_computer.get_unit_num(i); j++)
                        {
                            dis_b[i, j] = Math.Sqrt(dis_b[i, j] / main_list.Count());
                            ave_b[i, j] = ave_b[i, j] / FMS_NN.BATCH;
                        }
                    }

                    //優先度のばらつきを観察する用
                    string path;

                    path = Application.StartupPath;
                    if (!path.EndsWith("\\"))
                    {
                        path += "\\";
                    }
                    path = "priority";
                    string result_file = path + "\\result.dat";
                    File.Delete(result_file);


                    //学習の進行状況を確認するためシミュレーション
                    simulator.read_nn_con();
                    int selection_type = simulator.select_selection_type2();

                    //重みの格納
                    for (i = 1; i < FMS_NN.LAYOR_MAX; i++)
                    {

                        for (j = 0; j < FMS_NN.UNIT_MAX + 1; j++)
                        {
                            for (k = 0; k < FMS_NN.UNIT_MAX + 1; k++)
                            {
                                w2[i, j, k] = BP_weight_data[i, j, k];
                            }
                        }
                    }
                    if (epoch == 0)
                        all_input_rec = true;


                    simulator.nnt_run(selection_type);

                    //適用対象の問題に対して正規化，白色化を行う
                    if (epoch == 0)
                    {
                        data_normalize_for_apply();
                        //data_white_for_apply();
                        all_input_rec = false;
                    }

                    if (FMS_BASE.FILE_MAX == 1)
                    {
                        result[0] = simulator.get_performance_func(0);
                        Console.WriteLine(result);
                    }
                    else
                    {
                        result = simulator.host_computer.result_statistics_for_nn();
                    }

                    //最小値が更新される場合のみ重みを保存
                    if (epoch == 0 || result[0] < best_result)
                    {
                        save_BP_weight2();
                        best_result = result[0];
                    }

                    //評価値を保存
                    string update_result_file = file.FILENAME_NN_BP_UPDATE_RESULT;
                    update_result_file = update_result_file + ".dat";
                    fout = new StreamWriter(update_result_file, true, Encoding.GetEncoding("SHIFT_JIS"));
                    for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                    {
                        fout.Write(tr_gosa[i] + "\t");
                        fout.Write(test_gosa[i] + "\t");
                    }
                    fout.Write(abs_weight + "\t");
                    if (FMS_NN.answer_correct_rate == true)
                    {
                        fout.Write(correct1 + "\t");
                        fout.Write(true_conf1 + "\t");
                        fout.Write(correct2 + "\t");
                        fout.Write(true_conf2 + "\t");
                    }
                    fout.Write(result[0] + "\t");
                    fout.Write(result[1] + "\t");
                    fout.Write(result[2] + "\t");
                    fout.WriteLine(result[3]);

                    fout.Close();

                    if (epoch == 0 || epoch % FMS_NN.SPAN == 0)
                        //すべてのデータの誤差を出力する
                        write_out_noize(update_result_file);


                    Console.WriteLine();
                }//end of epoch


                //新しくファイルを作成しコピー
                DirectoryInfo dir = new DirectoryInfo(WeightFile + "weight");
                DirectoryInfo[] dirs = dir.GetDirectories();
                FileInfo[] files = dir.GetFiles();
                foreach (FileInfo file in files)
                {
                    string temppath = Path.Combine(WeightFile + "トライアル" + (trial + 1) + "回目", file.Name);
                    file.CopyTo(temppath, false);
                }

            }//end of trial


        }


        /******************************************************************************
        関数名：read_nn
        引  数：なし
        動  作：初期設定   ニューロの基本構成と重みデータの読み込み
        戻り値：なし
        ******************************************************************************/
        public void read_nn_con()
        {
            int i;
            int layor_max;
            int[] unit_num, unit_info;

            string fn_name;
            StreamReader fin;

            //initファイルからNNの構成を読み込む
            fn_name = file.FILENAME_NN_INIT;
            fin = new StreamReader(fn_name, Encoding.GetEncoding("Shift_JIS"));
            FileNextReader fnr = new FileNextReader(fin);

            //層数の読み込み
            layor_max = Int32.Parse(fnr.next());
            host_computer.set_layor_max(layor_max);

            int test = host_computer.get_layor_max();


            //各層のユニット数の読み込み
            unit_num = new int[layor_max];
            for (i = 0; i < layor_max; i++)
            {
                unit_num[i] = Int32.Parse(fnr.next());
                host_computer.set_unit_num(i, unit_num[i]);
            }

            //入力層に使う情報の読み込み
            //unit_info = new int[unit_num[0]];
            //for (i = 0; i < unit_num[0]; i++)
            //{
            //    unit_info[i] = Int32.Parse(fnr.next());
            //    host_computer.set_unit_info(i, unit_info[i]);
            //}
            fin.Close();

            return;
        }

        /******************************************************************************
        関数名：CreateNNWeight
        引  数：なし
        動  作：NNTでの初期重み集団を作成
        戻り値：なし
        ******************************************************************************/
        private void CreateNNWeight(int trial)
        {
            int i, j, k, he_d;
            double w_max, he, d, he_num;
            wrand = new Random(trial);
            random_box_muller = new RandomBoxMuller(trial);

            //重みを作成する
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {

                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    //heの初期値を使用する場合のみ使用
                    d = (host_computer.get_unit_num(i - 1));
                    he = (Math.Sqrt(6 / d)) * 10000;
                    he_d = (int)he;
                    //k=最大値はすべて閾値
                    for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        //一様乱数
                        if (FMS_NN.POPULATION_TYPE_BP == 1)
                        {
                            if (wrand.Next() % 2 == 1)
                            {
                                w_max = (double)(1.0 * FMS_NN.W_INIT);
                            }
                            else
                            {
                                w_max = (double)(-1.0 * FMS_NN.W_INIT);
                            }

                            BP_weight_data[i, j, k] = w_max * wrand.Next() / ((double)Int32.MaxValue + 1);

                        }
                        //正規分布
                        else if (FMS_NN.POPULATION_TYPE_BP == 2)
                        {
                            BP_weight_data[i, j, k] = random_box_muller.next(trial);

                        }
                        //heの初期値
                        else if (FMS_NN.POPULATION_TYPE_BP == 3)
                        {
                            he_num = wrand.Next(-he_d, he_d);
                            BP_weight_data[i, j, k] = he_num / 10000;


                        }
                        else
                        {
                            Console.WriteLine("個体の発生方法を指定してください");
                        }

                    }
                }
            }

        }

        /******************************************************************************
		関数名：nn_run
		引  数：教師番号
		動  作：ニューラルネットワークシステム実行:各ユニットの出力値を計算
        注意：重み[i,j,k-1]が他プログラムとの関係でi層jユニットからi-1層kユニット間の重みとしている
		戻り値：なし
		******************************************************************************/
        private void nn_run_2(int num)
        {
            TEA_SAMPLE s = min_list[num];

            //各層の出力値の計算
            for (s.i = 1; s.i < host_computer.get_layor_max(); s.i++)
            {
                for (s.j = 0; s.j < host_computer.get_unit_num(s.i); s.j++)
                {
                    //i層j番ユニットの出力＝総和(i-1層k番ユニットの出力*i-1層j番ユニットとk番ユニット間の重み)+重み[i,j,0]（閾値）
                    s.output[s.i, s.j] = BP_weight_data[s.i, s.j, 0];

                    for (s.k = 0; s.k < host_computer.get_unit_num(s.i - 1); s.k++)
                    {
                        s.output[s.i, s.j] += s.output[s.i - 1, s.k] * BP_weight_data[s.i, s.j, s.k + 1];
                    }
                    //活性化関数に入れる（retified_linear_functionで正規化線形関数を使用,logistic_funcでシグモイドを使用）,ただし出力は線形
                    if (s.i != host_computer.get_layor_max() - 1)
                        s.output[s.i, s.j] = FMS_UTIL.retified_linear_function(s.output[s.i, s.j], FMS_NN.ZI);
                    if (s.i == host_computer.get_layor_max() - 1)
                    {
                        s.output[s.i, s.j] = FMS_UTIL.logistic_func(s.output[s.i, s.j], FMS_NN.ZI);
                    }

                }

            }
            return;
        }
        /******************************************************************************
		関数名：nn_run
		引  数：教師番号
		動  作：ニューラルネットワークシステム実行:各ユニットの出力値を計算
        注意：重み[i,j,k-1]が他プログラムとの関係でi層jユニットからi-1層kユニット間の重みとしている
		戻り値：なし
		******************************************************************************/
        private void nn_run(int num, int layor)
        {
            TEA_SAMPLE s = min_list[num];

            for (s.j = 0; s.j < host_computer.get_unit_num(layor); s.j++)
            {
                //ドロップアウト判定：0でなければドロップアウト
                if (drop_unit[layor, s.j] == 0)
                {
                    //i層j番ユニットの出力＝総和(i-1層k番ユニットの出力*i-1層j番ユニットとk番ユニット間の重み)+重み[i,j,0]（閾値）
                    s.output[layor, s.j] = BP_weight_data[layor, s.j, 0];

                    for (s.k = 0; s.k < host_computer.get_unit_num(layor - 1); s.k++)
                    {
                        s.output[layor, s.j] += s.output[layor - 1, s.k] * BP_weight_data[layor, s.j, s.k + 1];
                    }
                    //活性化関数に入れる（retified_linear_functionで正規化線形関数を使用,logistic_funcでシグモイドを使用）,ただし出力は線形
                    if (layor != host_computer.get_layor_max() - 1)
                        s.output[layor, s.j] = FMS_UTIL.retified_linear_function(s.output[layor, s.j], FMS_NN.ZI);
                    if (layor == host_computer.get_layor_max() - 1)
                    {
                        s.output[layor, s.j] = FMS_UTIL.logistic_func(s.output[layor, s.j], FMS_NN.ZI);
                    }
                }
                //ドロップアウト
                else
                {
                    s.output[layor, s.j] = 0;
                }

            }

            return;
        }
        /****************************************************************************
  　　　　関数名：cal_delta
  　　　　引数  ：なし
 　　　　 動作  ：各ユニットのdeltaを計算する
  　　　　戻り値：出力ユニットの平均2乗誤差
　　　　****************************************************************************/
        private void cal_delta(int num)
        {
            TEA_SAMPLE s = min_list[num];

            //出力層を恒等写像とする
            for (s.k = 0; s.k < host_computer.get_unit_num(host_computer.get_layor_max() - 1); s.k++)
                s.delta[host_computer.get_layor_max() - 1, s.k] = (s.output[host_computer.get_layor_max() - 1, s.k] - s.target[s.k]);

            //各層のδの計算
            for (s.i = host_computer.get_layor_max() - 2; s.i > 0; s.i--)
            {
                //ドロップアウト判定
                if (drop_unit[s.i, s.j] == 0)
                {
                    for (s.k = 0; s.k < host_computer.get_unit_num(s.i); s.k++)
                    {

                        //ReLuの場合
                        if (s.output[s.i, s.k] >= 0)
                        {
                            s.delta[s.i, s.k] = 0;
                            for (s.j = 0; s.j < host_computer.get_unit_num(s.i + 1); s.j++)
                            {
                                //i層のk番基本δ＝総和（i+1層j番δ*i+1層j番ユニットからi層k番ユニット間の重み）
                                s.delta[s.i, s.k] += s.delta[s.i + 1, s.j] * BP_weight_data[s.i + 1, s.j, s.k + 1];
                            }
                        }
                        else
                        {
                            s.delta[s.i, s.k] = 0;
                        }
                        /*
                        //シグモイド
                        s.delta[s.i, s.k] = 0;
                        for (s.j = 0; s.j < host_computer.get_unit_num(s.i + 1); s.j++)
                        {
                            //i層のk番基本δ＝総和（i+1層j番δ*i+1層j番ユニットからi層k番ユニット間の重み）
                            s.delta[s.i, s.k] += s.delta[s.i + 1, s.j] * BP_weight_data[s.i + 1, s.j, s.k + 1];
                        }
                        s.delta[s.i, s.k] = s.delta[s.i, s.k] * (1 - s.output[s.i, s.k]) * s.output[s.i, s.k];
                        */
                    }
                }
                else
                {
                    s.delta[s.i, s.k] = 0;
                }
            }

            return;
        }

        /****************************************************************************
  　　　　関数名：cal_gradient
  　　　　引数  ：教師番号
 　　　　 動作  ：各ユニット間に存在する誤差関数の勾配を計算する
          注意  ：勾配も重みと同様i層jユニットからi-1層kユニット間の勾配が[i,j,k]
  　　　　戻り値：なし
　　　　****************************************************************************/
        private void cal_gradient(int num)
        {
            TEA_SAMPLE s = min_list[num];

            //ユニット間の勾配を計算する
            for (s.i = 1; s.i < host_computer.get_layor_max(); s.i++)
            {
                for (s.j = 0; s.j < host_computer.get_unit_num(s.i); s.j++)
                {
                    //閾値の勾配
                    s.grad[s.i, s.j, 0] = 1.0 * s.delta[s.i, s.j];
                    for (s.k = 0; s.k < host_computer.get_unit_num(s.i - 1); s.k++)
                    {
                        s.grad[s.i, s.j, s.k + 1] = s.output[s.i - 1, s.k] * s.delta[s.i, s.j];
                    }

                }
            }
        }

        /****************************************************************************
          関数名：cal_batch_gradient
          引数  ：なし
          動作  ：現バッチにおける各勾配の平均を算出
          戻り値：なし
        ****************************************************************************/
        private void cal_batch_gradient_2()
        {
            int i, j, k;

            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        //バッチ内のすべての勾配の平均を計算
                        ave_grad[i, j, k] = min_list.Average(obj => obj.grad[i, j, k]);
                    }
                }
            }

        }

        /*
        /****************************************************************************
          関数名：update_weight
          引数  ：なし
          動作  ：平均勾配を使用した重みの更新を行う(モーメンタム法）
          戻り値：なし
        ****************************************************************************/
        private void update_weight(double eps)
        {
            /*
            int i, j, k, m;
            double delta_weight,p;
            m = 0;
            p = FMS_NN.MOMENT;

            //ユニットの重みを更新する
            for (i = host_computer.get_layor_max() - 1; i > 0; i--)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    delta_weight = p * moment[m] - (1 - p) * eps * ave_grad[m];
                    BP_weight_data[i, j, 0] = (BP_weight_data[i, j, 0]+delta_weight);
                    moment[m] = delta_weight;
                    m++;

                    for (k = 1; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        delta_weight = p * moment[m] - (1 - p) * eps * ave_grad[m];
                        BP_weight_data[i, j, k] =( BP_weight_data[i, j, k] + delta_weight);
                        moment[m] = delta_weight;
                        m++;
                    }
                }
            }
            */
        }

        /****************************************************************************
          関数名：update_weight
          引数  ：なし
          動作  ：平均勾配を使用した重みの更新を行う(RMSprop）
          戻り値：なし
        ****************************************************************************/
        private void update_weight_4(double eps)
        {
            /*
            int i, j, k, m;
            double p,q,e, mov_ave,delta_weight;
            m = 0;

            q = 0.95;
            e = Math.Pow(10,-6);
            p = FMS_NN.MOMENT;

            //ユニットの重みを更新する
            for (i = host_computer.get_layor_max() - 1; i > 0; i--)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    mov_ave = q * rms[m] + (1 - q) * Math.Pow(ave_grad[m], 2);
                    delta_weight = eps / (Math.Sqrt(mov_ave + e)) * ave_grad[m];
                    delta_weight = p * moment[m] - (1 - p)* delta_weight;
                    BP_weight_data[i, j, 0] = BP_weight_data[i, j, 0] + delta_weight;
                    rms[m] = mov_ave;
                    moment[m] = delta_weight;
                    m++;

                    for (k = 1; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        mov_ave = q * rms[m] + (1 - q) * Math.Pow(ave_grad[m], 2);
                        delta_weight = eps / (Math.Sqrt(mov_ave + e)) * ave_grad[m];
                        delta_weight = p * moment[m] - (1 - p) * delta_weight;
                        BP_weight_data[i, j, k] = BP_weight_data[i, j, k] + delta_weight;
                        rms[m] = mov_ave;
                        moment[m] = delta_weight;
                        m++;
                    }
                }
            }
            */
        }
        /****************************************************************************
          関数名：update_weight
          引数  ：eps：学習率,t：更新回数（使用しない）
          動作  ：平均勾配を使用した重みの更新を行う(ベーシック）
          戻り値：なし
        ****************************************************************************/
        private void update_weight(double eps, int t)
        {
            int i, j, k;

            //ユニットの重みを更新する
            for (i = host_computer.get_layor_max() - 1; i > 0; i--)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        BP_weight_data[i, j, k] = BP_weight_data[i, j, k] - eps * ave_grad[i, j, k];
                    }
                }
            }


        }
        /****************************************************************************
           関数名：update_weight
           引数  ：なし
           動作  ：平均勾配を使用した重みの更新を行う(Adam）
           戻り値：なし
         ****************************************************************************/
        private void update_weight_2(double eps, int t)
        {

            int i, j, k;
            double p, q, e, mov_v, delta_weight, mov_m;

            p = 0.9;
            q = 0.999;
            e = Math.Pow(10, -8);

            //ユニットの重みを更新する
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    //ドロップアウトユニットであれば更新しない
                    if (drop_unit[i, j] == 0)
                    {
                        for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                        {
                            mov_m = p * moment[i, j, k] + (1 - p) * ave_grad[i, j, k];
                            moment[i, j, k] = mov_m;
                            mov_m = mov_m / (1 - Math.Pow(p, t));
                            mov_v = q * rms[i, j, k] + (1 - q) * Math.Pow(ave_grad[i, j, k], 2);
                            rms[i, j, k] = mov_v;
                            mov_v = mov_v / (1 - Math.Pow(q, t));
                            delta_weight = -eps * mov_m / (Math.Sqrt(mov_v + e));
                            //RAMDA*BP_weight_data[i, j, k]:L1正則化
                            BP_weight_data[i, j, k] = BP_weight_data[i, j, k] + delta_weight - eps * FMS_NN.RAMDA * BP_weight_data[i, j, k];

                        }
                    }

                }
            }

        }

        /******************************************************************************
       関数名：cal_nn_weight_abs_ave
       引  数：なし
       動  作：重みの絶対値平均を求める
       戻り値：WeightAbsAve
       ******************************************************************************/
        public double cal_nn_weight_abs_ave()
        {
            int i, j, k, count = 0;
            double weight;
            double WeightAbsAve = 0, WeightAbsSum = 0;

            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        count++;
                        weight = BP_weight_data[i, j, k];
                        WeightAbsSum += Math.Abs(weight);
                    }
                }
            }

            WeightAbsAve = WeightAbsSum / count;

            return WeightAbsAve;
        }

        /******************************************************************************
        関数名：write_out_noize
        引  数：なし
         動  作：すべての問題の誤差を出力する
        戻り値：なし
        ******************************************************************************/
        public void write_out_noize(string path)
        {
            int i = 0;
            System.IO.StreamWriter fout;
            string gosa_file = file.FILENAME_NN_BP_GOSA_RESULT;
            fout = new System.IO.StreamWriter(gosa_file, true, Encoding.GetEncoding("SHIFT_JIS"));

            for (i = 2; i < FMS_BASE.JOB_MAX + 1; i++)
            {
                fout.Write(rate_by_conf[0, 0, i] + "\t");
                fout.Write(rate_by_conf[0, 1, i] + "\t");
                fout.Write(rate_by_conf[1, 0, i] + "\t");
                fout.Write(rate_by_conf[1, 1, i] + "\r");
            }
            fout.Close();
        }

        /******************************************************************************
        関数名：integrate_data
        引  数：なし
        動  作：初期設定   教師データの統合
        戻り値：なし
        ******************************************************************************/
        public void integrate_data()
        {
            int i, j;

            //result_file1の読み込み
            for (i = 0; i < FMS_NN.INTEGRATE_NUM; i++)
                read_result(0, i);
            //result_file2の読み込み
            for (i = 0; i < FMS_NN.INTEGRATE_NUM; i++)
                read_result(1, i);


            //教師データの読み込み
            for (i = 0; i < FMS_NN.INTEGRATE_NUM; i++)
            {
                for (j = 0; j < FMS_NN.EACH_FILE_TEA_NUM; j++)
                {
                    if (inte_result[i, j, 0] < inte_result[i, j, 1])
                    {
                        read_tea(0, i, j);
                    }
                    else
                    {
                        read_tea(1, i, j);
                    }
                }
            }
            Console.WriteLine("読み込み完了");

            //教師データの書き出し
            for (i = 0; i < FMS_NN.INTEGRATE_NUM; i++)
            {
                for (j = 0; j < FMS_NN.EACH_FILE_TEA_NUM; j++)
                    record_integrate(j, i);
                Console.WriteLine(i + "番目ファイル書き込み中");
            }

            //result_fileの書き出し
            for (i = 0; i < FMS_NN.INTEGRATE_NUM; i++)
            {
                record_inte_result(i);
                Console.WriteLine(i + "番目結果書き込み中");
            }

            integrate_list.Clear();

            return;
        }

        /******************************************************************************
 　　　 関数名：read_result
        引  数：なし
        動  作:教師データの結果を読み込む
        戻り値：なし
        ******************************************************************************/
        public void read_result(int i, int j)
        {
            int k, l;
            string fn_name;
            StreamReader fin;

            //initファイルからNNの構成を読み込む
            fn_name = file.FILENAME_INTEGRATE_TEA;
            fn_name = fn_name + "result" + i + j + ".dat";
            fin = new StreamReader(fn_name, Encoding.GetEncoding("Shift_JIS"));
            FileNextReader fnr = new FileNextReader(fin);

            for (k = 0; k < FMS_NN.EACH_FILE_TEA_NUM; k++)
            {
                //納期遅れコスト以外読み飛ばす
                for (l = 0; l < 10; l++)
                    fnr.next();
                //納期遅れコストを格納
                inte_result[j, k, i] = Double.Parse(fnr.next());
            }

            fin.Close();

            return;
        }

        /******************************************************************************
        関数名：read_tea
        引  数：なし
        動  作:教師データの結果を読み込む
        戻り値：なし
        ******************************************************************************/
        public void read_tea(int i, int j, int k)
        {
            int l;
            string fn_name;
            StreamReader fin;
            StreamReader fstrIN;
            TEA_SAMPLE_MAIN s;

            //initファイルからNNの構成を読み込む
            fn_name = file.FILENAME_INTEGRATE_TEA;
            fn_name = fn_name + "tea" + i + "\\BPtea" + j + "\\tea" + k + ".dat";

            fstrIN = new StreamReader(fn_name, System.Text.Encoding.GetEncoding("shift_jis"), true);
            FileNextReader fnr = new FileNextReader(fstrIN);


            //ファイルの末尾まで読みこむ
            while (fstrIN.Peek() != -1)
            {
                s = new TEA_SAMPLE_MAIN();

                //データセット番号を記録
                s.set_num = j;
                s.file_num = k;

                for (l = 0; l < FMS_NN.IN_UNIT; l++)
                {
                    //入力情報の格納
                    s.input[l] = double.Parse(fnr.next());
                }
                //目標値情報の格納
                s.target[0] = double.Parse(fnr.next());
                integrate_list.Add(s);
            }

            fstrIN.Close();

            return;
        }

        /******************************************************************************
        関数名：record_integrate
        引  数：なし
        動  作：統合したデータを記録
        戻り値：なし
        ******************************************************************************/
        private void record_integrate(int jj, int k)
        {
            int i, j;
            file = new FMS_FILE();
            string f_name = file.FILENAME_TEA;
            string f2_name = file.FILENAME_TEA2;
            string fp_name;
            StreamWriter fstrOut;

            f2_name = f2_name + k + "\\";

            Directory.CreateDirectory(file.FILENAME_TEA_FILE);
            Directory.CreateDirectory(f2_name);


            fp_name = f2_name;
            IEnumerable<TEA_SAMPLE_MAIN> element = integrate_list.Where(obj => obj.set_num == k && obj.file_num == jj);
            integrate_sub_list.AddRange(element);

            f2_name = fp_name + "tea" + jj + ".dat";
            fstrOut = new StreamWriter(f2_name, false);
            //入出力情報の格納
            for (i = 0; i < integrate_sub_list.Count(); i++)
            {

                TEA_SAMPLE_MAIN s = integrate_sub_list[i];
                //入力情報
                for (j = 0; j < FMS_NN.IN_UNIT; j++)
                    fstrOut.Write(s.input[j] + "\t");
                //目標値
                fstrOut.WriteLine(s.target[0] + "\t");

            }

            fstrOut.Close();
            integrate_sub_list.Clear();

            return;
        }

        /******************************************************************************
        関数名：record_inte_result
        引  数：なし
        動  作：統合したデータを記録
        戻り値：なし
        ******************************************************************************/
        private void record_inte_result(int inte_num)
        {
            int i, j;
            file = new FMS_FILE();
            string f_name = file.FOLDANAME_RESULT;
            string fp_name;
            StreamWriter fstrOut;

            Directory.CreateDirectory(file.FOLDANAME_RESULT);

            // 教師ファイル名作成
            fp_name = f_name;
            fp_name += "\\result0" + inte_num + ".dat";
            fstrOut = new StreamWriter(fp_name, false);

            for (j = 0; j < FMS_NN.EACH_FILE_TEA_NUM; j++)
            {
                //納期遅れコストの小さいほうを記録
                if (inte_result[inte_num, j, 0] < inte_result[inte_num, j, 1])
                {
                    //次回統合用に重み付き納期遅れ時間以外を適当に埋めておく
                    for (i = 0; i < 10; i++)
                        fstrOut.Write(0 + "\t");

                    fstrOut.WriteLine(inte_result[inte_num, j, 0] + "\t");
                }
                else
                {
                    //次回統合用に重み付き納期遅れ時間以外を適当に埋めておく
                    for (i = 0; i < 10; i++)
                        fstrOut.Write(0 + "\t");
                    fstrOut.WriteLine(inte_result[inte_num, j, 1] + "\t");
                }
            }

            fstrOut.Close();

            return;
        }

        /******************************************************************************
		  関数名：read_tea_date
		  引  数：なし
		  動  作：教師データを読み込む
		  戻り値：なし
		******************************************************************************/
        private void read_tea_date()
        {
            int i, ii, j, k, tea_num, data_num;
            double conf;
            string num;
            file = new FMS_FILE();
            string f_name = file.FILENAME_TEA;
            string fp_name;
            k = 0;
            TEA_SAMPLE_MAIN s;
            tea_num = 0;
            conf = 0;
            j = 0;
            ii = 0;
            data_num = 0;
            StreamReader fstrIN;

            // 教師ファイル名
            fp_name = f_name;
            num = tea_num.ToString();
            fp_name += num + ".dat";

            do
            {
                fstrIN = new StreamReader(fp_name, System.Text.Encoding.GetEncoding("shift_jis"), true);
                FileNextReader fnr = new FileNextReader(fstrIN);


                //ファイルの末尾まで読みこむ
                while (fstrIN.Peek() != -1)
                {
                    s = new TEA_SAMPLE_MAIN();

                    //教師ファイル番号
                    s.file_num = tea_num;
                    //不要？
                    s.batch = k;
                    k++;

                    for (i = 0; i < FMS_NN.IN_UNIT; i++)
                    {
                        //入力情報の格納
                        s.input[i] = double.Parse(fnr.next());
                    }

                    //出力目標値の格納
                    for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                    {
                        s.target[i] = double.Parse(fnr.next());
                    }

                    //競合グループの番号(1～）
                    if (conf != s.input[0] || ii == conf)
                    {
                        ii = 0;
                        j++;
                        conf = s.input[0];
                    }
                    ii++;
                    s.conf_gr = j;

                    main_list.Add(s);

                    data_num++;
                }

                fstrIN.Close();

                tea_num++;
                fp_name = f_name;
                num = tea_num.ToString();
                fp_name += num + ".dat";

            } while (File.Exists(fp_name));
            Console.WriteLine("総データ数は" + data_num);
            return;
        }

        /******************************************************************************
		  関数名：remove_data
		  引  数：なし
		  動  作：教師データ数を削減する．教師データのうち一定数をテストデータにする．教師データ内からは削除する．
		  戻り値：なし
		******************************************************************************/
        private void remove_data()
        {
            int i, ii, index;
            double num;
            wrand = new Random(0);

            //教師データの削減
            for (i = 0; i <= main_list.Max(obj => obj.file_num); i++)
            {
                //各ファイルごとに一定数データを削除
                IEnumerable<TEA_SAMPLE_MAIN> element = main_list.Where(obj => obj.file_num == i);
                test_list.AddRange(element);

                for (ii = test_list.Count(); ii > FMS_NN.MAX_DATA; ii--)
                {
                    index = wrand.Next(0, test_list.Count());
                    test_list.RemoveAt(index);
                }
                main_list.RemoveAll(obj => obj.file_num == i);
                main_list.AddRange(test_list);
                test_list.Clear();
            }

            //ミニバッチ内の教師事例を入れ替える
            List<int> numbers = new List<int>();
            for (i = 0; i <= main_list.Max(obj => obj.conf_gr); i++)
            {
                numbers.Add(i);
            }
            num = (main_list.Max(obj => obj.conf_gr) * FMS_NN.TEST_RATE) / 10;
            if (FMS_NN.TEST_RATE > 0)
            {
                //テストデータ変換
                for (i = 1; i < num + 1; i++)
                {
                    index = wrand.Next(0, numbers.Count);
                    IEnumerable<TEA_SAMPLE_MAIN> element = main_list.Where(obj => obj.conf_gr == numbers[index]);
                    test_list.AddRange(element);
                    main_list.RemoveAll(obj => obj.conf_gr == numbers[index]);
                    numbers.RemoveAt(index);
                }
            }

            numbers.Clear();
            return;
        }

        /******************************************************************************
		  関数名：forming_data
		  引  数：なし
		  動  作：教師データの正規化等を行う予定
		  戻り値：なし
		******************************************************************************/
        private void forming_data()
        {
            int i;
            double max, min;

            //出力目標値を[0,1]にする（各教師事例を変えた時の比較用，異なるNNで教師を作った時は使わない）
            for (i = 0; i < FMS_NN.OUT_UNIT; i++)
            {
                min = list.Min(obj => obj.target[i]);
                list.ForEach(d =>
                {
                    if (min > 0)
                        d.target[i] = d.target[i] - min;
                    else if (min <= 0)
                        d.target[i] = d.target[i] + Math.Abs(min);
                });
                min_max[0, i] = min;
                nn_min_max[0, i] = min;
                max = list.Max(obj => obj.target[i]);
                list.ForEach(d =>
                {
                    d.target[i] = d.target[i] / max;
                });
                min_max[1, i] = max;
                nn_min_max[1, i] = max;
            }

            return;
        }

        /******************************************************************************
		  関数名：data_white
		  引  数：なし
		  動  作：データの白色化：ヤコビ法による固有ベクトルの算出
		  戻り値：なし
		******************************************************************************/
        private void data_white()
        {
            int s, t, i, j, max_s, max_t;
            double u;
            double ast = 1, bb = 1;
            double z = Math.Pow(10, -8);   //収束条件
            double[,] var_mat = new double[FMS_NN.IN_UNIT, FMS_NN.IN_UNIT]; //共分散行列
            double[,] var_mat2 = new double[FMS_NN.IN_UNIT, FMS_NN.IN_UNIT]; //共分散行列
            double[,] ort_mat = new double[FMS_NN.IN_UNIT, FMS_NN.IN_UNIT]; //直行行列
            double[,] eigen2 = new double[FMS_NN.IN_UNIT, FMS_NN.IN_UNIT]; //固有ベクトル
            double[] data_white = new double[FMS_NN.IN_UNIT]; //白色化用
            //全サンプルの共分散行列(1)の作成
            for (s = 0; s < FMS_NN.IN_UNIT; s++)
            {
                for (t = 0; t < FMS_NN.IN_UNIT; t++)
                {
                    var_mat[s, t] = main_list.Average(obj => obj.input[s] * obj.input[t]);
                }
            }

            for (s = 0; s < FMS_NN.IN_UNIT; s++)
            {
                for (t = 0; t < FMS_NN.IN_UNIT; t++)
                {
                    if (s == t)
                        eigen[s, t] = 1;
                    else
                        eigen[s, t] = 0;
                }
            }
            //ヤコビ法を用いた固有値，固有ベクトルの算出
            //共分散行列における上三角非対角成分(s>t)の中から絶対値最大成分|ast(k)|の抽出

            //収束判定値を満たすまで繰り返す
            while (Math.Abs(ast) > z)
            {
                ast = 0;
                max_s = max_t = 0;
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    for (t = 0; t < s; t++)
                    {
                        bb = Math.Abs(var_mat[s, t]);
                        if (ast < bb)
                        {
                            ast = bb;
                            max_s = s;
                            max_t = t;
                        }

                    }
                }
                //直行行列Pの作成:P=E:単位行列とする
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    for (t = 0; t < FMS_NN.IN_UNIT; t++)
                    {
                        if (s == t)
                            ort_mat[s, t] = 1;
                        else
                            ort_mat[s, t] = 0;
                    }
                }
                //θの決定
                if (var_mat[max_s, max_s] - var_mat[max_t, max_t] == 0)
                    u = Math.PI / 4;
                else
                    u = 0.5 * (Math.Atan(-2 * var_mat[max_s, max_t] / (var_mat[max_s, max_s] - var_mat[max_t, max_t])));

                //Pの変換:（max_s,max_t）に従う
                //Pss=cosθ，Pst=sinθ,Pts=-sinθ,Ptt=cosθ
                ort_mat[max_s, max_s] = Math.Cos(u);
                ort_mat[max_s, max_t] = Math.Sin(u);
                ort_mat[max_t, max_s] = -Math.Sin(u);
                ort_mat[max_t, max_t] = Math.Cos(u);

                //共分散行列(k+1)＝転置P×共分散行列(k)×Pとして更新
                //転置P×共分散行列：転置Pはmax_t,max_sの入れ替え
                ort_mat[max_s, max_t] = -Math.Sin(u);
                ort_mat[max_t, max_s] = Math.Sin(u);

                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    for (t = 0; t < FMS_NN.IN_UNIT; t++)
                    {
                        var_mat2[s, t] = 0;
                        for (i = 0; i < FMS_NN.IN_UNIT; i++)
                        {
                            var_mat2[s, t] = var_mat2[s, t] + (ort_mat[s, i] * var_mat[i, t]);
                        }
                    }
                }
                //×P
                ort_mat[max_s, max_t] = Math.Sin(u);
                ort_mat[max_t, max_s] = -Math.Sin(u);
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    for (t = 0; t < FMS_NN.IN_UNIT; t++)
                    {
                        var_mat[s, t] = 0;
                        for (i = 0; i < FMS_NN.IN_UNIT; i++)
                        {
                            var_mat[s, t] = var_mat[s, t] + (var_mat2[s, i] * ort_mat[i, t]);
                        }
                    }
                }

                //計算用に固有ベクトルを転写
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    for (t = 0; t < FMS_NN.IN_UNIT; t++)
                    {
                        eigen2[s, t] = eigen[s, t];
                    }
                }
                //X=XPとして固有ベクトルの保存
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    for (t = 0; t < FMS_NN.IN_UNIT; t++)
                    {
                        eigen[s, t] = 0;
                        for (i = 0; i < FMS_NN.IN_UNIT; i++)
                        {
                            eigen[s, t] = eigen[s, t] + (eigen2[s, i] * ort_mat[i, t]);
                        }
                    }
                }

                //共分散行列における上三角非対角成分(s>t)の中から絶対値最大成分|ast(k)|の抽出
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    for (t = 0; t < s; t++)
                    {
                        bb = var_mat[s, t];
                        if (ast < Math.Abs(bb))
                        {
                            ast = bb;
                            max_s = s;
                            max_t = t;
                        }
                    }
                }

            }
            //計算用に固有ベクトルを転写
            for (s = 0; s < FMS_NN.IN_UNIT; s++)
            {
                for (t = 0; t < FMS_NN.IN_UNIT; t++)
                {
                    eigen2[s, t] = eigen[s, t];
                }
            }
            //対角化行列の転置
            for (s = 0; s < FMS_NN.IN_UNIT; s++)
            {
                for (t = 0; t < FMS_NN.IN_UNIT; t++)
                {
                    eigen[s, t] = eigen2[t, s];
                }
            }


            //白色化：data(white)=転置X×data
            main_list.ForEach(d =>
            {
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    data_white[s] = 0;
                    for (i = 0; i < FMS_NN.IN_UNIT; i++)
                    {
                        data_white[s] = data_white[s] + (eigen[s, i] * d.input[i]);
                    }
                }
                for (s = 0; s < FMS_NN.IN_UNIT; s++)
                {
                    d.input[s] = data_white[s];
                }
            }
            );
            return;
        }

        /******************************************************************************
         関数名：date_normalize
         引  数：なし
         動  作：教師データを平均0分散１に正規化
         戻り値：なし
       ******************************************************************************/
        private void data_normalize()
        {
            int i, j;
            double num;
            double[] ave = new double[FMS_NN.IN_UNIT];
            double[] sum = new double[FMS_NN.IN_UNIT];
            double e = Math.Pow(10, -12);

            for (j = 0; j <= 0; j++)
                sum[j] = 0;
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
            {
                //データの平均を算出
                ave_test[i] = main_list.Average(obj => obj.input[i]);


                main_list.ForEach(d =>
                {
                    d.input[i] = d.input[i] - ave_test[i];
                });

                sum[i] = main_list.Sum(obj => Math.Pow(obj.input[i], 2));

                num = main_list.Count();

                dis_test[i] = Math.Sqrt((sum[i] + e) / num);

                main_list.ForEach(d =>
                {
                    d.input[i] = d.input[i] / dis_test[i];
                });
            }
        }

        /******************************************************************************
         関数名：date_normalize
         引  数：なし
         動  作：教師データを平均0分散１に正規化
         戻り値：なし
       ******************************************************************************/
        private void data_normalize2()
        {
            int i, j;
            double num;
            double[] ave = new double[FMS_NN.IN_UNIT];
            double[] sum = new double[FMS_NN.IN_UNIT];
            double e = Math.Pow(10, -12);

            for (j = 0; j <= 0; j++)
                sum[j] = 0;
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
            {
                //データの平均を算出
                ave_test2[i] = main_list.Average(obj => obj.input[i]);


                main_list.ForEach(d =>
                {
                    d.input[i] = d.input[i] - ave_test2[i];
                });

                sum[i] = main_list.Sum(obj => Math.Pow(obj.input[i], 2));

                num = main_list.Count();

                dis_test2[i] = Math.Sqrt((sum[i] + e) / num);

                main_list.ForEach(d =>
                {
                    d.input[i] = d.input[i] / dis_test2[i];
                });
            }
        }
        /******************************************************************************
        関数名：date_normalize_for_apply
        引  数：なし
        動  作：教師データを平均0分散１に正規化
        戻り値：なし
      ******************************************************************************/
        private void data_normalize_for_apply()
        {
            int i, j;
            double num;
            double[] ave = new double[FMS_NN.IN_UNIT];
            double[] sum = new double[FMS_NN.IN_UNIT];
            double e = Math.Pow(10, -12);
            num = all_input_data.Count();

            for (j = 0; j <= 0; j++)
                sum[j] = 0;
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
            {
                //データの平均を算出
                ave_test2[i] = all_input_data.Average(obj => obj.input[i]);

                sum[i] = all_input_data.Sum(obj => Math.Pow((obj.input[i] - ave_test2[i]), 2));

                dis_test2[i] = Math.Sqrt((sum[i] + e) / num);

            }
        }

        /******************************************************************************
         関数名：make_min_batch
         引  数：min_batch
         動  作：全教師データから一定数のデータを抜き取りミニバッチとする
         戻り値：なし
        ******************************************************************************/
        private void make_min_batch(int min_batch)
        {
            int i;
            TEA_SAMPLE s;
            //batch番号一致データをelementに抜き出す
            IEnumerable<TEA_SAMPLE_MAIN> element = main_list.Where(obj => obj.batch == min_batch);
            main_min_list.AddRange(element);

            main_min_list.ForEach(d =>
            {
                s = new TEA_SAMPLE();

                s.file_num = d.file_num;
                s.batch = d.batch;
                s.conf_gr = d.conf_gr;

                for (i = 0; i < FMS_NN.IN_UNIT; i++)
                    s.output[0, i] = d.input[i];

                for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                    s.target[i] = d.target[i];

                min_list.Add(s);
            });
            //移動用リストを削除する
            main_min_list.Clear();
            //抜き出したデータは削除する
            main_list.RemoveAll(obj => obj.batch == min_batch);


        }
        /******************************************************************************
         関数名：convert_list
         引  数：なし
         動  作：TEA_SAMPLE_MAINからTEA_SAMPLEへリストの変更
         戻り値：なし
        ******************************************************************************/
        private void convert_list()
        {
            int i;
            TEA_SAMPLE s;

            min_list.Clear();
            test_list.ForEach(d =>
            {
                s = new TEA_SAMPLE();

                s.file_num = d.file_num;
                s.batch = d.batch;
                s.conf_gr = d.conf_gr;

                for (i = 0; i < FMS_NN.IN_UNIT; i++)
                    s.output[0, i] = d.input[i];

                for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                    s.target[i] = d.target[i];

                min_list.Add(s);
            });

        }
        /******************************************************************************
         関数名：drop_out
         引  数：なし
         動  作：NNから外すユニットを決定する．そのユニットの出力とそのユニットへの逆伝搬を0とする
         戻り値：なし
        ******************************************************************************/
        private void drop_out()
        {
            int i, j;
            double p;

            //隠すユニットの選定
            for (i = 0; i < host_computer.get_layor_max() - 1; i++)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    //0:使用するユニット，1：使用しないユニット
                    drop_unit[i, j] = 0;
                    p = wrand.Next(1, 10);
                    //dropout使用時のみ
                    if (FMS_NN.DROP_OUT == true)
                    {
                        //入力層は20％の確率で隠す
                        if (i == 0 && p > 8)
                        {
                            drop_unit[i, j] = 1;
                        }
                        //中間層は50％の確率で隠す
                        else if (i > 1 && p > 5)
                        {
                            drop_unit[i, j] = 1;
                        }
                    }

                }
            }
        }
        /******************************************************************************
         関数名：return_batch_data
         引  数：min_batch
         動  作：ミニバッチのデータをすべてmain_listに返す
         戻り値：なし
        ******************************************************************************/
        private void return_batch_data()
        {
            int i;
            TEA_SAMPLE_MAIN s;

            min_list.ForEach(d =>
            {
                s = new TEA_SAMPLE_MAIN();

                s.file_num = d.file_num;
                s.batch = d.batch;
                s.conf_gr = d.conf_gr;

                for (i = 0; i < FMS_NN.IN_UNIT; i++)
                    s.input[i] = d.output[0, i];

                for (i = 0; i < FMS_NN.OUT_UNIT; i++)
                {
                    s.target[i] = d.target[i];
                    s.output[i] = d.output[FMS_NN.LAYOR_MAX - 1, i];
                }

                main_list.Add(s);
            });

            min_list.Clear();
        }
        /******************************************************************************
         関数名：batch_normalize
         引  数：なし
         動  作：バッチごとの各ユニットの出力値を平均0分散１に正規化
         戻り値：なし
        ******************************************************************************/
        private void batch_normalize(int layor)
        {
            int i, j;
            double num;
            double[] ave = new double[FMS_NN.UNIT_MAX];
            double[] dis = new double[FMS_NN.UNIT_MAX];
            double[] sum = new double[FMS_NN.UNIT_MAX];
            double e = Math.Pow(10, -10);

            for (j = 0; j <= 0; j++)
                sum[j] = 0;
            for (i = 0; i < host_computer.get_unit_num(layor); i++)
            {
                //データの平均を算出
                ave[i] = min_list.Average(obj => obj.output[layor, i]);

                min_list.ForEach(d =>
                {
                    d.output[layor, i] = d.output[layor, i] - ave[i];
                });

                sum[i] = min_list.Sum(obj => Math.Pow(obj.output[layor, i], 2));

                num = min_list.Count();

                dis[i] = Math.Sqrt((sum[i] + e) / num);

                min_list.ForEach(d =>
                {
                    d.output[layor, i] = d.output[layor, i] / dis[i];
                });

                //未学習問題用に平均分散を記録
                ave_b[layor, i] = ave_b[layor, i] + ave[i];
                dis_b[layor, i] = dis_b[layor, i] + sum[i];
            }

        }
        /******************************************************************************
       関数名：save_BP_weight
       引  数：なし
       動  作：NNTでの最良値を算出した重みを退避しておく
       戻り値：なし
       ******************************************************************************/
        public void save_BP_weight()
        {
            int i, j, k;
            string update_weight = file.FILENAME_NN_WEIGHTS_UPDATE;
            string update_weight_file;
            StreamWriter fout;

            update_weight_file = update_weight + "weight\\nn_weight.dat";
            fout = new StreamWriter(update_weight_file, false, Encoding.GetEncoding("SHIFT_JIS"));

            for (i = 0; i < host_computer.get_layor_max(); i++)
            {
                fout.Write(host_computer.get_unit_num(i) + "\t");
            }
            fout.WriteLine();

            //正規化処理で用いる各平均・分散
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
                fout.Write(ave_test[i] + "\t");
            fout.WriteLine();
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
                fout.Write(dis_test[i] + "\t");
            fout.WriteLine();

            //白色化：固有値ベクトル
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
            {
                fout.WriteLine();
                for (j = 0; j < FMS_NN.IN_UNIT; j++)
                {
                    fout.Write(eigen[i, j] + "\t");
                }
            }

            //バッチノーマライゼーションにおける各出力の平均・分散
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                fout.WriteLine();

                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    fout.Write(ave_b[i, j] + "\t");
                }
            }

            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                fout.WriteLine();

                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    fout.Write(dis_b[i, j] + "\t");
                }
            }

            //重み
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                fout.WriteLine();

                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        if (k == host_computer.get_unit_num(i - 1))
                        {
                            fout.WriteLine(BP_weight_data[i, j, k]);
                        }
                        else
                        {
                            fout.Write(BP_weight_data[i, j, k] + "\t");
                        }
                    }
                }
            }

            fout.Close();
        }

        /******************************************************************************
      関数名：save_BP_weight
      引  数：なし
      動  作：NNTでの最良値を算出した重みを退避しておく
      戻り値：なし
      ******************************************************************************/
        public void save_BP_weight2()
        {
            int i, j, k;
            string update_weight = file.FILENAME_NN_WEIGHTS_UPDATE;
            string update_weight_file;
            StreamWriter fout;

            update_weight_file = update_weight + "weight\\nn_weight.dat";
            fout = new StreamWriter(update_weight_file, false, Encoding.GetEncoding("SHIFT_JIS"));

            for (i = 0; i < host_computer.get_layor_max(); i++)
            {
                fout.Write(host_computer.get_unit_num(i) + "\t");
            }
            fout.WriteLine();

            //正規化処理で用いる各平均・分散
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
                fout.Write(ave_test2[i] + "\t");
            fout.WriteLine();
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
                fout.Write(dis_test2[i] + "\t");
            fout.WriteLine();

            //白色化：固有値ベクトル
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
            {
                fout.WriteLine();
                for (j = 0; j < FMS_NN.IN_UNIT; j++)
                {
                    fout.Write(eigen[i, j] + "\t");
                }
            }

            //バッチノーマライゼーションにおける各出力の平均・分散
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                fout.WriteLine();

                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    fout.Write(ave_b[i, j] + "\t");
                }
            }

            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                fout.WriteLine();

                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    fout.Write(dis_b[i, j] + "\t");
                }
            }

            //重み
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                fout.WriteLine();

                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        if (k == host_computer.get_unit_num(i - 1))
                        {
                            fout.WriteLine(BP_weight_data[i, j, k]);
                        }
                        else
                        {
                            fout.Write(BP_weight_data[i, j, k] + "\t");
                        }
                    }
                }
            }

            fout.Close();
        }
        /******************************************************************************
       関数名：read_nn_w
       引  数：なし
       動  作：初期設定   ニューロの基本構成と重みデータの読み込み
       戻り値：なし
       ******************************************************************************/
        public void read_nn_w()
        {
            int i, j, k;
            int UnitNum;
            string fw_name;
            StreamReader fin;

            //重みファイルからNN実行に使う重みを読み込む
            fw_name = file.FILENAME_NN_WEIGHT;
            fin = new StreamReader(fw_name, Encoding.GetEncoding("Shift_JIS"));
            FileNextReader fnr = new FileNextReader(fin);

            for (i = 0; i < host_computer.get_layor_max(); i++)
            {
                UnitNum = Int32.Parse(fnr.next());
                if (UnitNum != host_computer.get_unit_num(i)) { Console.WriteLine("第" + i + "層のユニット数が異なります"); }
            }

            for (i = 0; i < FMS_NN.IN_UNIT; i++)
                ave_test[i] = double.Parse(fnr.next());
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
                dis_test[i] = double.Parse(fnr.next());

            //白色化：固有値ベクトル
            for (i = 0; i < FMS_NN.IN_UNIT; i++)
            {
                for (j = 0; j < FMS_NN.IN_UNIT; j++)
                {
                    eigen[i, j] = double.Parse(fnr.next());//使ってない
                }
            }

            int test = host_computer.get_layor_max();

            //バッチノーマライゼーション：平均・分散
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    ave_b[i, j] = double.Parse(fnr.next());//評価用問題では使わない
                }
            }
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    dis_b[i, j] = double.Parse(fnr.next());//評価用問題では使わない
                }
            }

            //重みの格納
            for (i = 1; i < host_computer.get_layor_max(); i++)
            {
                for (j = 0; j < host_computer.get_unit_num(i); j++)
                {
                    for (k = 0; k < host_computer.get_unit_num(i - 1) + 1; k++)
                    {
                        w[i, j, k] = double.Parse(fnr.next());
                        host_computer.set_weight(i, j, k, w[i, j, k]);//使ってない
                    }
                }
            }
            fin.Close();

            return;
        }

    }



    public class TEA_SAMPLE     //教師データに対する出力，勾配，δの管理
    {
        //教師ファイル番号
        public int file_num { get; set; }

        //教師データのバッチ番号
        public int batch { get; set; }

        //競合グループ番号
        public int conf_gr { get; set; }

        //入力データ列
        public double[] input = new double[FMS_NN.UNIT_MAX];

        //出力データ配列
        public double[,] output = new double[FMS_NN.LAYOR_MAX, FMS_NN.UNIT_MAX + 1];

        //誤差データ配列
        public double[,] delta = new double[FMS_NN.LAYOR_MAX, FMS_NN.UNIT_MAX + 1];

        //勾配データ配列
        public double[,,] grad = new double[FMS_NN.LAYOR_MAX, FMS_NN.UNIT_MAX + 1, FMS_NN.UNIT_MAX + 1];

        //出力目標値
        //public double target { get; set; }

        //出力目標値
        public double[] target = new double[FMS_NN.OUT_UNIT];

        //カウンター:並列処理用
        public int i { get; set; }
        public int j { get; set; }
        public int k { get; set; }

    }
    public class TEA_SAMPLE_MAIN     //教師データに対する出力，勾配，δの管理
    {
        //データセット番号
        public int set_num { get; set; }

        //教師ファイル番号
        public int file_num { get; set; }

        //教師データのバッチ番号
        public int batch { get; set; }

        //競合グループ番号
        public int conf_gr { get; set; }

        //入力データ列
        public double[] input = new double[FMS_NN.IN_UNIT];

        //出力値
        public double[] output = new double[FMS_NN.OUT_UNIT];

        //出力目標値
        public double[] target = new double[FMS_NN.OUT_UNIT];

    }
    public class CONF_RECORD_DATA     //スケジュール時のデータ記録:競合ジョブ
    {
        //各ジョブの情報
        public double[] job_data = new double[FMS_NN.IN_UNIT + 4];

    }
    public class PARALLEL
    {
        //カウンター:並列処理用
        public int i { get; set; }
        public int j { get; set; }
        public int k { get; set; }
    }
}


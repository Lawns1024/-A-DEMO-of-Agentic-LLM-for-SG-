#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include <fstream>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("CeviIotMacSimulation");

// ==========================================
// 1. 全局统计变量与文件输出流
// ==========================================
uint64_t g_epochRxBytes = 0;
uint32_t g_epochAckedCount = 0;
uint32_t g_epochFailedCount = 0;

std::ofstream g_csvFile;
double g_collisionThreshold = 0.12;
double g_epochIntervalSeconds = 0.1;

// ==========================================
// 2. Trace 追踪回调函数
// ==========================================
void ServerRxCallback(Ptr<const Packet> p, const Address &addr) {
    g_epochRxBytes += p->GetSize();
}

void AckedMpduCallback(Ptr<const WifiMpdu> mpdu) {
    g_epochAckedCount++;
}

void MacDataFailedCallback(Mac48Address addr) {
    g_epochFailedCount++;
}

// ==========================================
// 3. 策略结构与动态注入机制
// ==========================================
struct Profile {
    uint32_t m_min;
    uint32_t m_max;
};

class PolicyMap {
public:
    Profile GetProfile(uint32_t nodeId, std::string state) {
        // CEVI策略：当系统拥挤时，节点依据相关均衡推荐概率错开竞争窗口
        if (state == "s_congested") {
            if (nodeId % 2 == 0) return {2, 2}; // CWmin = 31 (保守退避)
            else return {4, 4};                 // CWmin = 63 (高度保守)
        }
        return {1, 1}; // s_idle 状态下恢复基础乘子 (CWmin=15)
    }
};

PolicyMap globalPolicy;
NodeContainer staNodes;
uint32_t baseCwMin = 15;
uint32_t baseCwMax = 1023;

void InjectPolicy(Ptr<NetDevice> device, std::string state, PolicyMap& policy) {
    Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice>(device);
    Ptr<WifiMac> mac = wifiDevice->GetMac();
    uint32_t nodeId = wifiDevice->GetNode()->GetId();
    Profile prof = policy.GetProfile(nodeId, state);
    
    // 适配 802.11ax 的 QoS 队列机制
    PointerValue ptrVal;
    mac->GetAttribute("BE_Txop", ptrVal);
    Ptr<QosTxop> txQueue = ptrVal.Get<QosTxop>();
    
    txQueue->SetMinCw(prof.m_min * baseCwMin);
    txQueue->SetMaxCw(prof.m_max * baseCwMax);
}

// ==========================================
// 4. 博弈观测窗调度器 (100ms Epoch)
// ==========================================
void MonitorAndInject() {
    double interval = g_epochIntervalSeconds; // 博弈判定周期
    
    // 计算 AP 端在当前 Epoch 窗口的真实吞吐量 (Goodput)
    double throughputMbps = (g_epochRxBytes * 8.0) / (interval * 1e6);
    
    // 计算此 100ms 窗口内的 MAC 层碰撞率：failed / (acked + failed)
    uint32_t attemptsDelta = g_epochAckedCount + g_epochFailedCount;
    double collisionRate = 0.0;
    if (attemptsDelta > 0) {
        collisionRate = static_cast<double>(g_epochFailedCount) / static_cast<double>(attemptsDelta);
    }

    // 状态判定逻辑：基于碰撞率阈值 (如 15%)
    std::string currentState = (collisionRate > g_collisionThreshold)? "s_congested" : "s_idle";

    // 记录到 CSV 以供绘图
    double currentTime = Simulator::Now().GetSeconds();
    g_csvFile << currentTime << "," << throughputMbps << "," << collisionRate << "," << currentState << "\n";
    NS_LOG_INFO("Time: " << currentTime << "s | Goodput: " << throughputMbps << " Mbps | Collision: " << collisionRate << " | State: " << currentState);

    // 清零窗口计数，开始下一个 Epoch 的增量累积
    g_epochRxBytes = 0;
    g_epochAckedCount = 0;
    g_epochFailedCount = 0;

    // 向所有节点下发更新后的 CEVI 策略
    for (uint32_t i = 0; i < staNodes.GetN(); ++i) {
        InjectPolicy(staNodes.Get(i)->GetDevice(0), currentState, globalPolicy);
    }

    // 递归调度下一个 100ms 周期
    Simulator::Schedule(Seconds(g_epochIntervalSeconds), &MonitorAndInject);
}

int main(int argc, char *argv[]) {
    double simulationTime = 50.0;
    uint32_t payloadSize = 1024;
    uint32_t clientIntervalUs = 1500;
    double collisionThreshold = 0.12;
    
    CommandLine cmd;
    cmd.AddValue("clientIntervalUs", "UDP client send interval in microseconds", clientIntervalUs);
    cmd.AddValue("collisionThreshold", "Collision threshold for s_congested state", collisionThreshold);
    cmd.Parse(argc, argv);

    g_collisionThreshold = collisionThreshold;
    
    Time::SetResolution(Time::NS);
    LogComponentEnable("CeviIotMacSimulation", LOG_LEVEL_INFO);

    // 锁定物理层为 20MHz 窄带信道（ChannelWidth 需通过 ChannelSettings 生效）
    Config::SetDefault("ns3::WifiPhy::ChannelSettings", StringValue("{0,20,BAND_2_4GHZ,0}"));

    // 打开输出文件并写入表头
    g_csvFile.open("cevi_metrics.csv");
    g_csvFile << "Time_s,Throughput_Mbps,CollisionRate,State\n";

    NodeContainer apNode;
    apNode.Create(1);
    staNodes.Create(4);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetChannel(channel.Create());
    
    // 物理层阈值对齐
    phy.Set("CcaEdThreshold", DoubleValue(-82.0));
    phy.Set("RxSensitivity", DoubleValue(-92.0));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode", StringValue("HeMcs4"),
                                 "ControlMode", StringValue("HeMcs4"));

    WifiMacHelper mac;
    Ssid ssid = Ssid("IoT-Game-Theory-Network");
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    NetDeviceContainer staDevices = wifi.Install(phy, mac, staNodes);
    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice = wifi.Install(phy, mac, apNode);

    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0), "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(5.0), "DeltaY", DoubleValue(5.0),
                                  "GridWidth", UintegerValue(3), "LayoutType", StringValue("RowFirst"));
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(staNodes);
    mobility.Install(apNode);

    InternetStackHelper stack;
    stack.Install(apNode);
    stack.Install(staNodes);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staNodeInterfaces = address.Assign(staDevices);
    Ipv4InterfaceContainer apNodeInterface = address.Assign(apDevice);

    uint16_t port = 9;
    UdpServerHelper server(port);
    ApplicationContainer serverApp = server.Install(apNode.Get(0));
    serverApp.Start(Seconds(0.0));
    serverApp.Stop(Seconds(simulationTime + 1.0));

    for (uint32_t i = 0; i < staNodes.GetN(); ++i) {
        UdpClientHelper client(apNodeInterface.GetAddress(0), port);
        client.SetAttribute("MaxPackets", UintegerValue(32000000));
    client.SetAttribute("Interval", TimeValue(MicroSeconds(clientIntervalUs))); 
        client.SetAttribute("PacketSize", UintegerValue(payloadSize));
        ApplicationContainer clientApp = client.Install(staNodes.Get(i));
        clientApp.Start(Seconds(1.0 + i * 0.01)); 
        clientApp.Stop(Seconds(simulationTime));
    }

    // 绑定追踪探针 (需在应用层装载后进行)
    Config::ConnectWithoutContext("/NodeList/0/ApplicationList/0/$ns3::UdpServer/Rx", MakeCallback(&ServerRxCallback));
    
    // 为 4 个 STA (NodeList 1 到 4) 绑定底层 MAC 层 ACK 成功与 ACK 失败追踪
    for (uint32_t i = 1; i <= 4; ++i) {
        std::ostringstream ackedMpduPath, macDataFailedPath;
        ackedMpduPath << "/NodeList/" << i << "/DeviceList/0/$ns3::WifiNetDevice/Mac/AckedMpdu";
        Config::ConnectWithoutContext(ackedMpduPath.str(), MakeCallback(&AckedMpduCallback));
        
        macDataFailedPath << "/NodeList/" << i
                          << "/DeviceList/0/$ns3::WifiNetDevice/RemoteStationManager/MacTxDataFailed";
        Config::ConnectWithoutContext(macDataFailedPath.str(), MakeCallback(&MacDataFailedCallback));
    }

    // 跳过网络启动不稳定期，在 1.5 秒清空窗口计数，并在 1.6 秒开始第一次真正观测
    Simulator::Schedule(Seconds(1.5), []() {
        g_epochRxBytes = 0;
        g_epochAckedCount = 0;
        g_epochFailedCount = 0;
        Simulator::Schedule(MilliSeconds(100), &MonitorAndInject);
    });

    NS_LOG_INFO("Starting CEVI MAC Simulation for " << simulationTime << " seconds...");
    Simulator::Stop(Seconds(simulationTime + 0.1));
    Simulator::Run();
    Simulator::Destroy();
    
    g_csvFile.close();
    return 0;
}
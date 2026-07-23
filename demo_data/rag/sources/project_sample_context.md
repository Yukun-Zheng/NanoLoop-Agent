# NanoLoop 项目样品标签与材料上下文

## 这张知识卡解决什么问题

NanoLoop 当前数据中出现的 `LaCo`、`LaCr`、`LaCu`、`LaMn`、`LaNi`、`NdCo`、`NdCu`、`NdNi` 是项目内部的**样品组标签**。它们表示样品中关注的 A 位元素族（La 或 Nd）与过渡金属元素族（Co、Cr、Cu、Mn 或 Ni），但**不能仅凭标签推出完整化学式、掺杂比例、氧非化学计量、晶相或制备条件**。

## 问答时必须遵守的边界

- 用户只提供 `LaNi` 时，可以回答“这是 La/Ni 相关样品组”，但不能自动改写成 `LaNiO3`、`La2NiO4` 或其他确定化学式。
- 讨论材料性质前，应优先读取上传图像绑定的 `material_name`、`material_formula`、样品编号、处理气氛、温度、时间和比例尺。
- 如果完整化学式缺失，应把回答限定为“相关钙钛矿或复合氧化物体系的一般规律”，并明确这不是对当前样品的成分确认。
- 图像分割得到的颗粒数量、粒径、数密度和覆盖率是当前样品的实验数据；文献中的机理、用途和材料性质是外部知识，两者必须分区展示。
- 仅凭 SEM 形貌不能确认颗粒元素组成、价态、晶相或析出机理。需要 EDS、XPS、XRD、TEM 等证据才能进一步确认。

## 推荐的材料上下文字段

- `sample_id`
- `material_name`
- `material_formula`
- `material_aliases`
- `a_site_elements`
- `b_site_elements`
- `dopant_fraction`
- `oxygen_nonstoichiometry`
- `treatment_atmosphere`
- `treatment_temperature_c`
- `treatment_time_h`
- `microscope_and_scale`

## 适用的项目标签

LaCo、LaCr、LaCu、LaMn、LaNi、NdCo、NdCu、NdNi。

## 来源与说明

本卡是 NanoLoop Agent 项目知识资产，用于约束样品标签解释和防止问答系统把内部简称误当作完整化学式。它不替代材料配方记录或实验原始台账。

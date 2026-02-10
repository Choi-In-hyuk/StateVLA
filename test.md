graph TD
    %% 전체 스타일 설정
    classDef enc fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef latent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef action fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef predictor fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    %% ---------------------------------------------------------
    %% Phase 1: JEPA (World Model 학습)
    %% ---------------------------------------------------------
    subgraph Phase_1_Representation_Learning [Step 1: JEPA 사전 학습 (Representation)]
        direction TB
        
        %% 입력 데이터
        Img_t(현재 이미지 Xt) --> Vision_Enc[Vision Encoder]:::enc
        Text_L(언어 명령어 L) --> Text_Enc[Text Encoder]:::enc
        
        %% 인코더 융합 -> 현재 상태 Zt
        Vision_Enc & Text_Enc --> Z_t((현재 상태 임베딩 Zt)):::latent
        
        %% 예측기 (Predictor): 여기가 핵심!
        Action_t(로봇 액션 At) --> Predictor[Predictor / World Model]:::predictor
        Z_t --> Predictor
        
        %% 미래 예측
        Predictor --> Z_pred(예측된 미래 Z't+1):::latent
        
        %% 정답지 (Target)
        Img_next(다음 이미지 Xt+1) --> Target_Enc[Target Encoder / EMA]:::enc
        Target_Enc --> Z_true((실제 미래 Zt+1)):::latent
        
        %% Loss
        Z_pred <-->|JEPA Loss: 최소화| Z_true
    end

    %% ---------------------------------------------------------
    %% Phase 2: Flow Matching (Policy 학습)
    %% ---------------------------------------------------------
    subgraph Phase_2_Policy_Learning [Step 2: Flow Matching 정책 학습 (Action)]
        direction TB
        
        %% 조건 입력 (Phase 1에서 학습된 인코더 고정)
        New_Img(입력 이미지) --> Frozen_V[Frozen Encoder]:::enc
        New_Text(입력 명령어) --> Frozen_T[Frozen Encoder]:::enc
        
        Frozen_V & Frozen_T --> Z_cond((조건 임베딩 Zt)):::latent
        
        %% Flow Matching 헤드
        Noise(가우시안 노이즈 X0) --> FM_Head[Flow Matching Policy Head]:::action
        Z_cond -->|Conditioning| FM_Head
        Time(Time Step t) --> FM_Head
        
        %% 최종 액션 생성
        FM_Head -->|ODE Solver / Denoising| Final_Action(연속적인 액션 궤적 Output):::action
    end
    
    %% 연결선 스타일
    linkStyle default stroke-width:2px,fill:none,stroke:#333;

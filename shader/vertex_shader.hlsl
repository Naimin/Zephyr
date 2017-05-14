struct VS_INPUT
{
	float4 pos : POSITION;
	float4 normal : NORMAL;
	float4 color: COLOR;
	float2 texCoord : TEXCOORD;
};

struct VS_OUTPUT
{
	float4 pos: SV_POSITION;
	float2 texCoord: TEXCOORD;
};

cbuffer CB : register(b0)
{
	float4x4 wvpMat;
};

VS_OUTPUT main(VS_INPUT input)
{
	VS_OUTPUT output;
	output.pos = mul(input.pos, wvpMat);
	output.texCoord = input.texCoord;

	return output;
}